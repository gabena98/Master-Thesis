import polars as pl
from polars import col
import random
import numpy as np
import torch
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#### PARAMETERS

mimic_prefix  = '../../data/physionet.org/files/mimiciv/3.1/hosp'
ccs_prefix    = '../data'
output_prefix = '../data/processed'

# these are the mimic-iv tables
diagnoses_file  = 'diagnoses_icd.csv.gz' 
admissions_file = 'admissions.csv.gz'
# this is to convert icd10 to icd8
icd_conv_file   = '../data/diagnosis_gems_2018/2018_I10gem.txt'
# this is for converting icd9 to ccs
ccs_single_file = '../data/Single_Level_CCS_2015/icd9_to_ccs_dx2015.csv'
# this contains the ontology over the icds
ontology_file   = '../data/ICD9CM.csv'

# these are the output files
output_diagnoses  = 'diagnoses.parquet'
output_ccs        = 'ccs.parquet'
output_icd        = 'icd.parquet'
output_ontology   = 'ontology.parquet'
output_generation = 'generation.parquet'

# minimum number of icd occurrences to not be assigned a 0 code
min_icd_occurences = 0
# this is for the generator model
num_output_codes = 500 

train_fraction = 0.8
eval_fraction  = 0.1

#### END PARAMETERS

ontology_prefixes = ['http://purl.bioontology.org/ontology/ICD9CM/', 'http://purl.bioontology.org/ontology/STY/']
ontology_root_name = 'root'

# load all the data
diagnoses_path = os.path.join(mimic_prefix, diagnoses_file)
diagnoses      = pl.read_csv(diagnoses_path, schema_overrides={'subject_id':pl.UInt64, 'icd_version':pl.UInt8})
admission_path = os.path.join(mimic_prefix, admissions_file)
admissions     = pl.read_csv(admission_path)
icd_conv_path  = os.path.join(ccs_prefix, icd_conv_file)
icd_conv       = pl.read_csv(icd_conv_path, schema_overrides={'icd10cm':pl.Utf8, 'icd9cm':pl.Utf8})
ccs_conv_path  = os.path.join(ccs_prefix, ccs_single_file)
ccs_conv       = pl.read_csv(ccs_conv_path, quote_char="'")

diagnoses = diagnoses[['subject_id', 'hadm_id', 'icd_code', 'icd_version']]
diagnoses = diagnoses.with_columns (
    visit_count = col('hadm_id').unique().count().over('subject_id')
).filter(col('visit_count') > 1)

icd_conv = icd_conv.lazy().select (
    col('icd10cm'),
    col('icd9cm').first().over('icd10cm')
).unique().collect()

# convert icd10 to icd9
diagnoses_icd10 = (
    diagnoses
        .filter(col('icd_version') == 10)
        .join(icd_conv, left_on='icd_code', right_on='icd10cm', how='left')
        .select(col('subject_id'), col('hadm_id'), col('icd9cm').alias('icd_code'))
)
diagnoses_icd9 = diagnoses.filter(col('icd_version') == 9)[diagnoses_icd10.columns]
diagnoses = pl.concat([diagnoses_icd9, diagnoses_icd10], how='vertical')

# convert icd9 to ccs
ccs_conv = ccs_conv.select (
    icd9 = col('ICD-9-CM CODE').str.strip_chars(),
    ccs  = col('CCS CATEGORY' ).str.strip_chars().cast(pl.UInt16),
    description = col('CCS CATEGORY DESCRIPTION')
)
diagnoses = (
    diagnoses
    .join(
        ccs_conv[['icd9', 'ccs']],
        left_on  = 'icd_code',
        right_on = 'icd9',
        how = 'left'
    )
    .with_columns (
        ccs      = col('ccs'     ).fill_null(pl.lit(0)),
        icd_code = col('icd_code').fill_null(pl.lit('NoDx'))
    )
).unique()

# convert ccs codes to ids
ccs_codes = diagnoses[['ccs']].unique().sort('ccs')
we_have_zero_code = 0 in ccs_codes['ccs'].head(1)
if not we_have_zero_code:
    print('WARNING: we do not have a zero code!')
ccs_codes = ccs_codes.join(ccs_conv.drop('icd9').unique('ccs'), on='ccs', how='left')
starting_index = 0 if we_have_zero_code else 1
indexes = pl.DataFrame({'ccs_id': range(starting_index, starting_index+ccs_codes.shape[0])}, schema={'ccs_id':pl.UInt32})
ccs_codes = pl.concat([ccs_codes, indexes], how='horizontal')
diagnoses = diagnoses.join(ccs_codes[['ccs', 'ccs_id']], on='ccs', how='left').drop('ccs')

# convert icd9 codes to id
icd9_codes = diagnoses['icd_code'].value_counts()
num_codes_before = icd9_codes.shape[0]
icd9_codes = icd9_codes.filter(col('count') >= min_icd_occurences).drop('count')
print(
    f'There were {num_codes_before} different icd codes in the dataset. '
    f'We are keeping only those with at least {min_icd_occurences}. '
    f'There are {icd9_codes.shape[0]} icd codes remaining'
)
indexes = pl.DataFrame({'icd9_id': range(1, icd9_codes.shape[0]+1)}, schema={'icd9_id':pl.UInt32})
icd9_codes = pl.concat([icd9_codes, indexes], how='horizontal')
icd9_codes = icd9_codes.with_columns(icd9_id=pl.when(pl.col('icd_code') == 'NoDx').then(pl.lit(0)).otherwise(pl.col('icd9_id')))
diagnoses = (
    diagnoses
    .join (
        icd9_codes,
        on = 'icd_code',
        how = 'left',
    )
    .with_columns(icd9_id=col('icd9_id').fill_null(pl.lit(0)))
    .drop('icd_code')
)

# add time data
admissions = admissions.select (
    col('hadm_id'),
    col('admittime').str.to_datetime('%Y-%m-%d %H:%M:%S')
)
diagnoses = diagnoses.join(admissions, on='hadm_id', how='left').drop('hadm_id')


# this is for the generative model
# take the top 500 most frequent codes
top_codes = diagnoses['icd9_id'].value_counts().sort('count', descending=True).head(num_output_codes)
# remove the NoDx code
top_codes = top_codes.filter(pl.col('icd9_id') != 0)
top_codes = top_codes.select(
    pl.col('icd9_id'),
    out_id = pl.arange(1, top_codes.shape[0]+1),
)
diagnoses = diagnoses.join(top_codes, on='icd9_id', how='left')
diagnoses = diagnoses.with_columns(pl.col('out_id').fill_null(0))
generation_conversion = top_codes.join(icd9_codes, on='icd9_id')
generation_conversion = generation_conversion.join(ccs_conv, left_on='icd_code', right_on='icd9', how='left')
generation_conversion = generation_conversion.join(ccs_codes, on='ccs', how='left')
generation_conversion = generation_conversion[['icd9_id', 'out_id', 'ccs_id']]


# prepare for use
diagnoses_a = (
    diagnoses
    .with_columns(
        position = col('admittime').rank('dense').over('subject_id') - 1,
    )
    .group_by('subject_id')
    .agg(
        col(['ccs_id', 'icd9_id', 'position', 'out_id']).sort_by('admittime'),
    )
)
diagnoses_b = (
    diagnoses
    .group_by(['subject_id', 'admittime'])
    .agg(
        pl.len().alias('count'),
     )
    .group_by('subject_id')
    .agg(pl.col('count').sort_by('admittime'))
)
diagnoses = diagnoses_a.join(diagnoses_b, on='subject_id', how='inner').sort('subject_id')

# Split in train/eval/test

data_size = len(diagnoses)
ind = np.random.permutation(data_size)
n_train = int(train_fraction * data_size)
n_valid = int(eval_fraction * data_size)
train_indices = ind[:n_train]
valid_indices = ind[n_train:(n_train + n_valid)]
test_indices = ind[(n_train + n_valid):]

diagnoses = diagnoses.with_columns(
    role = pl.when(pl.arange(0, data_size).is_in(train_indices)).then(pl.lit('train'))
          .when(pl.arange(0, data_size).is_in(valid_indices)).then(pl.lit('eval'))
          .otherwise(pl.lit('test'))
)

# Build Ontology

ontology   = pl.read_csv(ontology_file)

def remove_prefixes(exp: pl.Expr, prefixes: list[str]) -> pl.Expr:
    for p in prefixes:
        exp = exp.str.strip_prefix(p)
    return exp
ontology = ontology.lazy().select(
    label    = pl.col('Preferred Label'),
    icd_code = remove_prefixes(pl.col('Class ID'), ontology_prefixes),
    parent   = remove_prefixes(pl.col('Parents' ), ontology_prefixes),
).collect()

diagnoses_type = ontology.filter(pl.col('parent').str.starts_with('http') & (pl.col('label') != 'PROCEDURES'))['icd_code']

num_rows = 0
while num_rows != len(diagnoses_type):
    num_rows = len(diagnoses_type)
    t = pl.DataFrame({'parent': diagnoses_type})
    t = t.join(ontology[['parent', 'icd_code']], how='left', on='parent')
    diagnoses_type = pl.concat([t['icd_code'], diagnoses_type]).unique()
t = pl.DataFrame({'icd_code': diagnoses_type})
ontology = t.join(ontology, how='left', on='icd_code')

ontology = ontology.with_columns(
    icd_code = pl.col('icd_code').str.replace('.', '', literal=True),
    parent   = pl.when (
        pl.col('parent').str.starts_with('http') 
    )
    .then(pl.lit(ontology_root_name))
    .otherwise(pl.col('parent').str.replace('.', '', literal=True).fill_null(pl.lit(ontology_root_name))),
)

ontology = ontology.join(icd9_codes, on='icd_code', how='full').filter(~pl.col('icd_code').is_null())

# this assigns root as parent of NoDx
ontology = ontology.with_columns(pl.col('parent').fill_null(pl.lit(ontology_root_name)))

uncoded = ontology.filter(pl.col('icd9_id').is_null())
first_new_id = icd9_codes['icd9_id'].max() + 1
indexes = pl.DataFrame({'icd9_id': range(first_new_id, first_new_id+len(uncoded))}, schema={'icd9_id':pl.UInt32})
uncoded = pl.concat([uncoded.drop('icd9_id'), indexes], how='horizontal')

root_id = first_new_id + len(uncoded)
# schema is {'icd_code': Utf8, 'label': Utf8, 'parent': Utf8, 'icd9_id': UInt32}
root_row = pl.DataFrame(
    {'icd_code':ontology_root_name, 'label':'Root of Ontology', 'parent':'root', 'icd_code_right': None , 'icd9_id':root_id},
    schema = ontology.schema,
)

ontology = pl.concat([
    ontology.filter(~pl.col('icd9_id').is_null()).sort('icd9_id'),
    uncoded,
    root_row,
])

dictionary = ontology.select(parent=pl.col('icd_code'), parent_id=pl.col('icd9_id'))
ontology = ontology.join(dictionary, how='left', on='parent')

# Save all the files

diagnoses_path  = os.path.join(output_prefix, output_diagnoses)
ccs_path        = os.path.join(output_prefix, output_ccs)
icd9_path       = os.path.join(output_prefix, output_icd)
ontology_path   = os.path.join(output_prefix, output_ontology)
generation_path = os.path.join(output_prefix, output_generation)

# Create directories if they do not exist
os.makedirs(os.path.dirname(diagnoses_path), exist_ok=True)
os.makedirs(os.path.dirname(ccs_path), exist_ok=True)
os.makedirs(os.path.dirname(icd9_path), exist_ok=True)
os.makedirs(os.path.dirname(ontology_path), exist_ok=True)
os.makedirs(os.path.dirname(generation_path), exist_ok=True)

print('')
print(f'diagnoses path is:  {diagnoses_path}')
print(f'ccs path is:        {ccs_path}')
print(f'icd9 path is:       {icd9_path}')
print(f'ontology path is:   {ontology_path}')
print(f'generative path is: {generation_path}')

# Save the DataFrames
diagnoses.write_parquet(diagnoses_path)
ccs_codes.write_parquet(ccs_path)
icd9_codes.write_parquet(icd9_path)
ontology.write_parquet(ontology_path)
generation_conversion.write_parquet(generation_path)