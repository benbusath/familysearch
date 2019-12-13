# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:04:20 2019

Created by Ben Busath
"""

import pandas as pd
import os
import sys
sys.path.append(r'V:\Python')
from FamilySearch import FamilySearch


def add_family(df,username,password):
	'''
	add_family takes in an island census dataframe, removes any households who
	has members attatched to tree, adds the head of household from the remaining 
	households to the tree, and returns a dataframe of tree-extending hints for
	the head arks and the newly created pids.
    
	Attributes:
		df (dataframe or infile):  	island census in dataframe format, or infile pointing to csv of island census
									island census must have arkid and hhid column. See documentation for more information
									
		username (str):				FamilySearch username
		password (str):				FamilySearch password
				
	Returns: 
		dataframe in the following format-
		ark{year} | pid | url
    '''
	if type(df)==str:
		df=pd.read_csv(df)
	
	fs=FamilySearch(username,password,os.getcwd(),'in.csv','out.csv')
	
	if  'ark1900' in df.columns.values:
        arkyear='ark1900'
    elif 'ark1910' in df.columns.values:
        arkyear='ark1910'
    elif 'ark1920' in df.columns.values:
        arkyear='ark1920'
    else:
        raise Exception('Ark column in input dataframe must be labeled in the format ark + year! (ex: ark1900)')
	
    to_add = check_for_existing(df)
    to_add = to_add[to_add['pr_relationship_to_head']=='Head'].drop_duplicates(subset='hhid')
    to_add.to_csv('in.csv',index=True,index_label='index')
    
    names_dict={
            'pr_name_gn' : 'given',
            'pr_name_surn' : 'surname',
            'pr_sex_code' : 'gender',
            'pr_birth_date' : 'birthdate',
            'pr_birth_place' : 'birthplace',
            'event_place' : 'resplace',
            'event_date' : 'resdate'
            }
    
    fs.AddPerson(names=names_dict)
    
    pid_index=pd.read_csv('out.csv')
    to_add=pd.read_csv('in.csv')
    to_add.rename(columns= {arkyear : 'ark'})
    results=to_add[[arkyear, 'index']].merge(pid_index, on='index')
    
    results['url']='https://www.familysearch.org/search/linker?pal=/ark:/61903/1:1:'+results[arkyear]+'&id='+results['pid']
    results=results[[arkyear,'pid','url']]
    os.remove('in.csv')
    
    #os.remove('out.csv')
    return results

# takes in dataframe with arks and drops households with person already on tree
    # appends potential hints to tree if already existing
def check_for_existing(df):
'''
takes in dataframe with arks and drops households with person already on tree
TO IMPLEMENT: appends potential hints to tree if already existing

Atrributes:
	df (dataframe): dataframe with ark and houshold_id column
	
Returns:
	df (dataframe): original dataframe but with households with people on tree dropped
'''
    df[[arkyear]].to_csv('in.csv',index=False)
    fs.GetPidFromArk(ark_col=0)
    out=pd.read_csv('out.csv').rename(columns={'ark':arkyear})
    os.remove('in.csv')
    os.remove('out.csv')
    
    df=df.merge(out, on=arkyear, how='outer')
    hhid_to_drop=df[df['pid'].notnull()]['hhid']
    new_df=df[~df['hhid'].isin(hhid_to_drop)]
    return new_df

    # FIXME add potential hints to list
    '''
    if len(new_df)<len(df):
        old_df=df[df['hhid'].isin(hhid_to_drop)]
        missing_someone=old_df[old_df['pid'].isnull()]['hhid']
        hints_to_add=old_df[(old_df['hhid'].isin(missing_someone))&(old_df['pid'].notnull())].drop_duplicates(subset='hhid')[[arkyear,pid]]
        hints_to_add[arkyear].to_list()
        hints_to_add['pid'].to_list()
    '''
    
if __name__ == '__main__':
    county = 'interracial_1910'
    
    os.chdir(r'R:\JoePriceResearch\record_linking\projects\african_american\interracial_couples\data\interracial_1910')
    df=pd.read_csv(county + '_island_census.csv')
    lol=add_family(df)
    lol.to_csv(r'R:\JoePriceResearch\record_linking\projects\african_american\interracial_couples\data\interracial_1910\created_hints.csv',index=False)
