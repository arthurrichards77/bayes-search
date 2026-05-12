"""
bn.py

Simple Bayesian network handling

"""

import pandas as pd
from math import prod

def validate_pdf_size(pdf):
    assert len(pdf)==prod([(len(pdf.drop('prob',axis=1).drop_duplicates(col))) for col in pdf.drop('prob',axis=1).columns])

def validate_pdf(pdf, tol=1e-10, all_combinations=False):
    if all_combinations:
        validate_pdf_size(pdf)
    total_prob = sum(pdf['prob'])
    assert abs(total_prob-1.0)<tol, f'Probability sums to {total_prob}'

def validate_cpdf(cpdf, all_combinations=True):
    if all_combinations:
        validate_pdf_size(cpdf)
    givens = [c for c in cpdf.columns if c.startswith('|')]
    for i,r in cpdf[givens].drop_duplicates().iterrows():
        assert sum(cpdf[cpdf.eq(r)[givens].all(axis=1)]['prob'])==1

def joint_pdf_independent(pdf1,pdf2,prune=True):
    jpdf = pd.merge(pdf1,pdf2,how='cross')
    jpdf['prob'] = jpdf['prob_x']*jpdf['prob_y']
    if prune==True:
        jpdf = jpdf[jpdf['prob']>0.0]
    return jpdf.drop(['prob_x','prob_y'],axis=1)

def joint_pdf_dependent(pdf,cpdf,prune=True):
    validate_cpdf(cpdf)
    validate_pdf(pdf)
    givens = [c for c in cpdf.columns if c.startswith('|')]
    givens_nobar = [col.strip('|') for col in givens]
    assert all([col in pdf.columns for col in givens_nobar])
    new_vars = [c for c in cpdf.drop('prob',axis=1).columns if not c.startswith('|')]
    jpdf1 = pd.merge(cpdf[new_vars].drop_duplicates(),pdf,how='cross')
    jpdf2 = pd.merge(jpdf1,cpdf.rename(columns=dict(zip(givens,givens_nobar))),on=givens_nobar+new_vars)
    jpdf2['prob'] = jpdf2['prob_x']*jpdf2['prob_y']
    if prune==True:
        jpdf2 = jpdf2[jpdf2['prob']>0.0]
    return jpdf2.drop(['prob_x','prob_y'],axis=1)

def get_marginal_pdf(jpdf,of):
    validate_pdf(jpdf)
    assert all([c in jpdf.columns for c in of])
    margin_cols = [c for c in jpdf.columns if (c!='prob') and (c not in of)]
    #print(margin_cols)
    pivot_jpdf = jpdf.pivot(index=of,columns=margin_cols,values='prob')
    return pivot_jpdf.sum(axis=1).reset_index().rename(columns={0:'prob'})

def get_conditional_pdf(jpdf,of,given):
    validate_pdf(jpdf)
    assert all([c in jpdf.columns for c in of])
    assert all([c in jpdf.columns for c in given])
    jmpdf1 = get_marginal_pdf(jpdf,of + given)
    jmpdf2 = get_marginal_pdf(jmpdf1,given)
    cpdf1 = pd.merge(jmpdf1,jmpdf2,on=given)
    cpdf1['prob'] = cpdf1['prob_x']/cpdf1['prob_y']
    renaming = dict([(col,'|'+col) for col in given])
    return cpdf1.drop(['prob_x','prob_y'],axis=1).rename(columns=renaming)
