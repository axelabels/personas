'Reconstruct the sociodemographic wording used in the study prompts.'


from typing import Any, Dict, Optional


import math


import re


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return True
        if value < 0:
            return True
    if isinstance(value, str) and value.strip().lower() in {'', 'nan', 'na', 'none'}:
        return True
    return False


def _pick(mapping: Dict[Any, str], value: Any, default: Optional[str]=None) -> Optional[str]:
    if _is_missing(value):
        return None
    return mapping.get(value, mapping.get(int(float(value)), default if default is not None else str(value)))


SEX_MAP = {1: 'male', 2: 'female'}


EDUCATION_X025_MAP = {1: 'with incomplete elementary education',
 2: 'with completed elementary education',
 3: 'with incomplete secondary vocational education',
 4: 'with completed secondary vocational education',
 5: 'with incomplete academic secondary education',
 6: 'with completed academic secondary education',
 7: 'with some university education but no degree',
 8: 'with a university degree'}


EDUCATION_X025A_MAP = {0: 'with no formal education',
 1: 'with primary education',
 2: 'with lower secondary education',
 3: 'with upper secondary education',
 4: 'with post-secondary non-tertiary education',
 5: 'with short-cycle tertiary education',
 6: 'with a university degree'}


EDUCATION_X025A_01_MAP = {0: 'with no formal education',
 1: 'with primary education',
 2: 'with lower secondary education',
 3: 'with upper secondary education',
 4: 'with post-secondary non-tertiary education',
 5: 'with short-cycle tertiary education',
 6: "with a bachelor's degree or equivalent",
 7: "with a master's degree or equivalent",
 8: 'with a doctoral degree or equivalent'}


EDUCATION_X025R_MAP = {1: 'with relatively low education',
 2: 'with relatively medium education',
 3: 'with relatively high education'}


EMPLOYMENT_MAP = {1: 'working full-time',
 2: 'working part-time',
 3: 'self-employed',
 4: 'retired',
 5: 'a homemaker',
 6: 'a student',
 7: 'unemployed',
 8: 'otherwise not in the labor force'}


INCOME_MAP = {1: 'in a low-income household',
 2: 'in a lower-middle income household',
 3: 'in a middle-income household'}


MARITAL_STATUS_MAP = {1: 'married',
 2: 'living with a partner',
 3: 'divorced',
 4: 'separated',
 5: 'widowed',
 6: 'single'}


URBAN_RURAL_MAP = {1: 'living in an urban area', 2: 'living in a rural area'}


SOCIAL_CLASS_MAP = {1: 'who sees themself as upper class',
 2: 'who sees themself as upper middle class',
 3: 'who sees themself as lower middle class',
 4: 'who sees themself as working class',
 5: 'who sees themself as lower class'}


def age_descriptor(age: Any) -> Optional[str]:
    if _is_missing(age):
        return None
    try:
        age_int = int(float(age))
        if age_int < 0 or age_int > 120:
            return None
        return f'{age_int} years old'
    except (TypeError, ValueError):
        return None


def household_descriptor(household_size: Any) -> Optional[str]:
    if _is_missing(household_size):
        return None
    try:
        n = int(float(household_size))
    except (TypeError, ValueError):
        return None
    if n == 1:
        return 'in a single-person household'
    return f'in a {n}-person household'


def _normalize_country_specific_education(value: Any) -> Optional[str]:
    """
    Last-resort formatter for country-specific education fields.
    If numeric, returns a generic label.
    If text, lightly cleans and returns it.
    """
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        try:
            n = int(value)
            return f'with country-specific education category {n}'
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub('\\s+', ' ', text)
    return f"with education level '{text}'"


def education_descriptor(row: Dict[str, Any]) -> Optional[str]:
    for col, mapping in [('X025', EDUCATION_X025_MAP), ('X025A_01', EDUCATION_X025A_01_MAP), ('X025A', EDUCATION_X025A_MAP), ('X025R', EDUCATION_X025R_MAP)]:
        desc = _pick(mapping, row.get(col))
        if desc:
            return desc
    for col in ('X025CSWVS', 'X025CSEVS'):
        desc = _normalize_country_specific_education(row.get(col))
        if desc:
            return desc
    return None


def sociodemographic_persona(row: Dict[str, Any]) -> str:
    """
    Transform WVS-style coded values into a compact natural-language descriptor.

    Expected keys:
      X003       age
      X001       sex
      X025       education
      X028       employment
      X047R_WVS  income level (3 categories)
      X045       subjective social class
      X007       marital status
      X013       household size
      X050C      urban/rural
    """
    parts = []
    age = age_descriptor(row.get('X003'))
    sex = _pick(SEX_MAP, row.get('X001'))
    if age and sex:
        parts.append(f'a {age} {sex}')
    elif age:
        parts.append(f'a {age}')
    elif sex:
        parts.append(f'a {sex}')
    education = education_descriptor(row)
    if education:
        parts.append(education)
    employment = _pick(EMPLOYMENT_MAP, row.get('X028'))
    if employment:
        parts.append(employment)
    # EVS and WVS store the same three-category income scale in separate fields.
    if _is_missing(row.get('X047R_WVS')):
        income = _pick(INCOME_MAP, row.get('X047R_EVS'))
    else:
        income = _pick(INCOME_MAP, row.get('X047R_WVS'))
    if income:
        parts.append(income)
    social_class = _pick(SOCIAL_CLASS_MAP, row.get('X045'))
    if social_class:
        parts.append(social_class)
    marital = _pick(MARITAL_STATUS_MAP, row.get('X007'))
    if marital:
        parts.append(marital)
    # The study's historical prompts did not include the children fields.
    household = household_descriptor(row.get('X013'))
    if household:
        parts.append(household)
    habitat = _pick(URBAN_RURAL_MAP, row.get('X050C'))
    if habitat:
        parts.append(habitat)
    if not parts:
        raise ValueError('No sociodemographic values available for this respondent')
    first, *rest = parts
    if not rest:
        return first.capitalize() + '.'
    return first.capitalize() + ', ' + ', '.join(rest) + '.'
