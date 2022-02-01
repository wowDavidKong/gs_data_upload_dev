"""
input_workbook_preparation.py

This script consolidates and prepares input data from a set of input templates into a workbook that can be read by the
optimiser scripts.
"""

import logging
import typing
import pandas as pd
import numpy as np
import re
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime
from sys import platform
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
info = logger.info
error = logger.error
warning = logger.warning

StringList = typing.List[str]

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the column names of a dataframe so there are no commonly used special characters in them.

    Args:
        df (pd.DataFrame): input dataframe with columns that need cleaning
    
    Returns:
        pd.DataFrame: output dataframe with cleaned columns
    """
    df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(' ', '_')
            .str.replace('(', '')
            .str.replace(')', '')
            .str.replace('.', '')
            .str.replace('/', '_or_')
            .str.replace('#','no')
            .str.replace('%','pc')
        )

    return df

def loadClient():
    if platform == "win32":
        key_path = r"C:\Users\1162723\Optimiser_Keys\dev_key.json"
        credentials = service_account.Credentials.from_service_account_file(
            key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    else:
        client = bigquery.Client(project='gcp-wow-pvc-grnstck-dev')

    return client

def bqUploader(name,dataframe,scenario):

    client = loadClient()

    df = dataframe.copy()

    if name in ['yield_trees','supplementary_primal_costs','supplemental_primal_limits','steer_limits','primal_sales_values','demand','allocation_side_costs','abattoir_limits','cattle_costs','primary_processing_costs']:

        # rename columns for date only columns
        if name in ["primary_processing_costs", "cattle_costs"]:
            columnList = list(df.columns.values)
            colDict = {}
            for col in columnList:
                if col != 'cattle_type':
                    colDict[col] = 'cost_per_head_'+col
                else:
                    colDict[col] = col
            df = df.rename(columns=colDict)

        # generate column names

        columnList = list(df.columns.values)
        regexDate = '[0-9]{4}-[0-9]{2}-[0-9]{2}'
        workingDates = []
        workingCols = []
        uniqueColumns = ["scenario_name", "date", "species"]
        
        for c in columnList:
            if not(re.search(regexDate,c)):
                if c not in uniqueColumns:
                    uniqueColumns.append(c)
                if c not in workingCols:
                    workingCols.append(c)
            r = re.findall(regexDate, c)

            if(len(r) > 0):
                c_dte = r[0]
                if(c_dte not in workingDates):
                    workingDates.append(c_dte)
                c_str = c.replace("_"+c_dte, "")
                c_str = c_str.replace(c_dte, "")
                if(c_str not in uniqueColumns):
                    uniqueColumns.append(c_str)
                if c_str not in workingCols:
                    if c_str != "":
                        workingCols.append(c_str)
        if("" in uniqueColumns):
            uniqueColumns.remove("")
        forDF = []
        for i, row in df.iterrows():
            for wD in workingDates:
                insertCol = [scenario]
                insertCol.append(wD)
                insertCol.append("beef")
                for wC in workingCols:
                    if(wC in ["abattoir", "primal_name", "cattle_type", "primal", "primal_type", "cattle_specification", "yield_tree", "beef_type"]):
                        insertCol.append(row[wC])
                    else:
                        insertCol.append(row[wC+"_"+wD])
                forDF.append(insertCol)

        # print(workingCols)
        exportDF = pd.DataFrame(forDF, columns = uniqueColumns)
        exportDF['rank'] = np.arange(exportDF.shape[0])
        table_id = "Optimiser.beef_" + name
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
        )

        job = client.load_table_from_dataframe(
            exportDF, table_id, job_config=job_config
        )

    else:
        print('not uploading ' + name)

class WorkbookPrep:
    """
    Workbook prep contains all the functions to load in the different input templates. They can be called individually
    and each will update a particular page or set of pages in the input workbook.
    """
    
    def __init__(self, file_dict, weeks, scenario):
        # self, workbook_loc: str, config: str
        """
        Loads in the config and the workbook template and initalises any data containers as well as some mappings.

        Args:
            config (str): file location for config
        """
        self.file_dict = file_dict
        self.weeks = weeks
        self.scenario = scenario
        self.esb_state_list =   ['SA_Retail_Items', 'NT_Retail_Items', 'QLD_Retail_Items', 'NSW_Retail_Items', 'VIC_Retail_Items', 'TAS_Retail_Items']
        self.asp = True
        self.df_cost_group_map = self.create_costgroups_map()

        # load in the sku to primal map
        self.df_sku_to_primal = (clean_column_names(pd.read_excel(self.file_dict['hilton_yield_data'], skiprows = 1))
                                .rename(columns = {'wow_article_no':'wow_retail_no','wow_primal_ref_no':'primal_article'})
                                )[['wow_retail_no','primal_article']]
        self.df_sku_to_primal['primal_article'] = self.df_sku_to_primal['primal_article'].str.replace(" ","")

        self.df_updated_demand = pd.DataFrame([])
        self.df_hgpf_demand = pd.DataFrame([])
        self.df_yield_tree_data = pd.DataFrame([])
        self.df_cattle_cost = pd.DataFrame([])
        self.cattle_cost_raw_data = pd.DataFrame([])
        self.df_processing_cost = pd.DataFrame([])
        self.df_abattoir_limits = pd.DataFrame([])
        self.df_steer_limits = pd.DataFrame([])
        self.df_steer_weights = pd.DataFrame([])
        self.df_supl_primal_costs = pd.DataFrame([])
        self.df_supl_primal_limits = pd.DataFrame([])
        self.df_updated_allocation_side_costs = pd.DataFrame([])
        self.df_updated_primal_sales_values = pd.DataFrame([])
        self.cattle_types = []


    def scenario_manager(self):
        client = loadClient()
        df = (client.query(f'SELECT * FROM `gcp-wow-pvc-grnstck-dev.Optimiser.scenario_species_list` WHERE scenario = "{self.scenario}" ').to_dataframe())

        scenarioList = df['scenario'].tolist()

        if self.scenario in scenarioList:
            print(f"deleting scenario {self.scenario}")
            sql = f"""
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.HGF_acc_makesheet` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.MSA_acc_makesheet` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.MSA_bunbury_makesheet` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.MSA_longford_makesheet` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.MSA_naracoorte_makesheet` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.MSA_tamworth_makesheet` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.PR_tamworth_makesheet` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.created_primal_split` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.grsfed_tamworth_makesheet` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.optimised_supply_purchases` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.primal_fallout` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser_Results.results_summary` where scenario = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser.beef_abattoir_limits` where scenario_name = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser.beef_allocation_side_costs` where scenario_name = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser.beef_cattle_costs` where scenario_name = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser.beef_demand` where scenario_name = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser.beef_primal_sales_values` where scenario_name = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser.beef_primary_processing_costs` where scenario_name = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser.beef_steer_limits` where scenario_name = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser.beef_supplemental_primal_limits` where scenario_name = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser.beef_supplementary_primal_costs` where scenario_name = "{self.scenario}";
                delete from `gcp-wow-pvc-grnstck-dev.Optimiser.beef_yield_trees` where scenario_name = "{self.scenario}";
                """
            query_job = client.query(sql) 
            query_job.result() 
        else:
            table_id = "Optimiser.scenario_species_list"
            dte = datetime.today().strftime('%Y-%m-%d')
            rows = [{u"run_date":dte, u"user":"dkong1@woolworths.com.au", u"scenario":self.scenario, u"species":"beef"}]

            client.insert_rows_json(
                table_id, rows, row_ids=[None] * len(rows)
            )
        return True


    def create_costgroups_map(self) -> dict:
        '''
        Create a mapping from primal to cost group.

        Returns:
            dict: the mapping from primal to cost group
        '''
        df_cost_groups_mapping = None

        try:
            client = loadClient()
            df_cost_groups_mapping =  (client.query(f'SELECT * FROM `gcp-wow-pvc-grnstck-dev.Optimiser.beef_article_list`').to_dataframe())
        except FileNotFoundError:
            print('Could not find the cost group mapping file in basic_cost_groups in file_dict please upload')
            self.logger.warning('Could not find the cost group mapping file in basic_cost_groups in file_dict please upload')
        if df_cost_groups_mapping is not None:
            try:
                df_cost_groups_mapping = df_cost_groups_mapping[['Product_Code', 'Product_Group_1']]
            except KeyError:
                print('Expecting 2 columns in the cost group mapping file: Product Code and Product Group 1. At least one of these is not there. Please make sure the basic_cost_groups file has the right format.')
                self.logger.warning('Expecting 2 columns in the cost group mapping file: Product Code and Product Group 1. At least one of these is not there. Please make sure the basic_cost_groups file has the right format.')
                exit()
            df_cost_groups_mapping.rename(columns={'Product_Code': 'product_code', 'Product_Group_1': 'cost_group'}, inplace=True)
            df_cost_groups_mapping['product_code'] = pd.to_numeric(df_cost_groups_mapping['product_code'], errors='coerce')
        
        # print(df_cost_groups_mapping)

        return df_cost_groups_mapping

   


    def load_yield_tree_template(self):
        """
        Loads a yield tree template.
        """
        info("Updating yield tree inputs")
        client = loadClient()
        self.df_yield_tree_data = (client.query(f'SELECT * FROM `gcp-wow-pvc-grnstck-dev.Optimiser.beef_yield_tree_list`').to_dataframe())
        self.df_yield_tree_data.rename(columns={'abattoir': 'abattoir', 'cattle_specification': 'cattle_specification', 'yield_tree_version': 'yield_tree'}, inplace=True)
        # Don't do anything with this yet, we wait until we have the steer weights and push it into the input workbook then



    def read_in_yield_tree_tab(self, file_path, sheet_name):
        """
        Reads in a sheet from a yield tree template

        Args:
              file_path (str): the path including the file name of the yield tree template excel workbook
              sheet_name (str): the name of the sheet to read in

        Returns:
              pd.DataFrame: a dataframe containing the yield tree specifications
        """

        # Isolate the data we're interested in
        df_yield_tree = pd.read_excel(file_path, sheet_name=sheet_name)
        df_yield_tree.drop(['Unnamed: 0'], axis=1, inplace=True)
        df_yield_tree.columns = df_yield_tree.iloc[4]
        if (df_yield_tree.columns.values != ['Abattoir', 'Cattle specification', 'Yield tree version']).all():
            warning(f"WARNING: Layout of yield template not as expected, please do not insert additional rows or columns or modify headers, terminating!")
            exit()
        df_yield_tree.drop(df_yield_tree.index[0:5], inplace=True)

        return df_yield_tree



    def load_abattoir_template(self):
        """
        Load in cattle prices and weights, cattle availability limits, abattoir processing limits and costs from a
        template.
        """

        info("Updating abattoir and cattle inputs")
        df_raw_abattoir_data_acc = self.read_in_abattoir_tab(self.file_dict['abattoir_template_loc'], 'ACC', 'acc')
        df_raw_abattoir_data_bunbury = self.read_in_abattoir_tab(self.file_dict['abattoir_template_loc'], 'Bunbury', 'bunbury')
        df_raw_abattoir_data_longford = self.read_in_abattoir_tab(self.file_dict['abattoir_template_loc'], 'Longford', 'longford')
        df_raw_abattoir_data_naracoorte = self.read_in_abattoir_tab(self.file_dict['abattoir_template_loc'], 'Teys Naracoorte', 'naracoorte')
        df_raw_abattoir_data_tamworth = self.read_in_abattoir_tab(self.file_dict['abattoir_template_loc'], 'Teys Tamworth', 'tamworth')

        df_combined_raw_abattoir_data = df_raw_abattoir_data_acc.append(df_raw_abattoir_data_bunbury)
        df_combined_raw_abattoir_data = df_combined_raw_abattoir_data.append(df_raw_abattoir_data_longford)
        df_combined_raw_abattoir_data = df_combined_raw_abattoir_data.append(df_raw_abattoir_data_naracoorte)
        df_combined_raw_abattoir_data = df_combined_raw_abattoir_data.append(df_raw_abattoir_data_tamworth)

        self.process_steer_costs(df_combined_raw_abattoir_data)
        self.process_processing_cost(df_combined_raw_abattoir_data)
        self.process_steer_limits(df_combined_raw_abattoir_data)
        self.process_abattoir_limits(df_combined_raw_abattoir_data)
        self.process_steer_weights(df_combined_raw_abattoir_data)



    def process_steer_costs(self, df_combined_raw_abattoir_data):
        """
        Extracts the steer costs out of the raw abattoir data and writes to the input workbook.

        Args:
            df_combined_raw_abattoir_data (pd.DataFrame): The raw abattoir related data that required further processing
        """
        # Convert some of the data into a cattle_type tag the optimiser uses
        mask = (df_combined_raw_abattoir_data['parameter'] == 'Avg steer cost') & ~(
                    df_combined_raw_abattoir_data['beef_type'] == 'Total')
        df_cattle_cost_data = df_combined_raw_abattoir_data.loc[mask, :].copy()

        df_cattle_cost_data.loc[:, 'cattle'] = 'MSA'
        df_cattle_cost_data.loc[df_cattle_cost_data['beef_type'] == 'HGF Grain-fed', 'cattle'] = 'HGF'
        df_cattle_cost_data.loc[df_cattle_cost_data['beef_type'] == 'grsfed', 'cattle'] = 'grsfed'
        df_cattle_cost_data.loc[df_cattle_cost_data['beef_type'] == 'PR', 'cattle'] = 'PR'
        df_cattle_cost_data.loc[df_cattle_cost_data['beef_type'] == 'Vealer', 'cattle'] = 'Vealer'
        df_cattle_cost_data.loc[df_cattle_cost_data['beef_type'] == 'grsfed Supplementary', 'cattle'] = 'grsfed'

        df_cattle_cost_data.loc[:, 'contract'] = 'contract_cattle'
        df_cattle_cost_data.loc[
            df_cattle_cost_data['beef_type'] == 'MSA Supplementary', 'contract'] = 'spot_purchase_cattle'

        df_cattle_cost_data.loc[
            df_cattle_cost_data['beef_type'] == 'grsfed Supplementary', 'contract'] = 'spot_purchase_cattle'

        df_cattle_cost_data.loc[:, 'cattle_type'] = df_cattle_cost_data['cattle'] + '_' + df_cattle_cost_data['abattoir'] + '_' + df_cattle_cost_data['contract']

        # Store a copy to use the cattle types we just created again for the processing costs
        df_cattle_cost_data = df_cattle_cost_data.dropna()
        self.cattle_cost_raw_data = df_cattle_cost_data.copy()

        df_cattle_cost_data = df_cattle_cost_data.drop(['beef_type', 'parameter', 'cattle', 'contract'], axis=1)

        # Now pivot back to a weekly format
        self.df_cattle_cost = (
            pd.pivot_table(df_cattle_cost_data,
                           values='value',
                           index='cattle_type',
                           columns='week',
                           aggfunc='sum')
                .reset_index()
        )

        # Report any missing values
        missing_cattle_cost_rows = self.df_cattle_cost[self.df_cattle_cost.isnull().any(axis=1)]
        if not missing_cattle_cost_rows.empty:
            for i, row in missing_cattle_cost_rows.iterrows():
                parts = row['cattle_type'].split('_')
                steer = 'contract'
                if 'spot' in parts:
                    steer = 'spot'

                warning(f"WARNING: Cattle cost data not provided for all weeks for {parts[0]} {steer} steer at abattoir {parts[1]}, terminating!")
            exit()
        bqUploader('cattle_costs',self.df_cattle_cost, self.scenario)
        # self.append_df_to_excel(self.workbook_loc, self.df_cattle_cost, 'cattle_costs', 0, True)



    def process_processing_cost(self, df_combined_raw_abattoir_data):
        """
        Extracts the abattoir limits out of the raw abattoir data and writes to the input workbook.

        Args:
            df_combined_raw_abattoir_data (pd.DataFrame): The raw abattoir related data that required further processing
        """

        # Pull out the cattle cost df which has a nice list of the different cattle types that we wil reuse
        df_cattle_cost_data = self.cattle_cost_raw_data

        # Create a df for the cattle processing costs as well
        mask = (df_combined_raw_abattoir_data['parameter'] == 'Processing cost (excluding product specific cost)')
        df_processing_cost_data = df_combined_raw_abattoir_data.loc[mask, :].copy()
        df_processing_cost_data = df_processing_cost_data.dropna()
        df_processing_cost_data = df_processing_cost_data.merge(df_cattle_cost_data, how='outer',
                                                                left_on=['abattoir', 'week', 'beef_type'],
                                                                right_on=['abattoir', 'week', 'cattle'],
                                                                suffixes=[None, '_cattle'])

        # Check for missing data after merge and warn user of any gaps
        df_missing_data = df_processing_cost_data[df_processing_cost_data.isnull().any(axis=1)]
        # print(df_missing_data)
        df_missing_data= df_missing_data.loc[df_missing_data['beef_type'] != 'Total']
        if not df_missing_data.empty:
            for i, row in df_missing_data.iterrows():
                if pd.isnull(row['value_cattle']):
                    warning(f"WARNING: There are abattoir processing costs cattle type {row['beef_type']} at abattoir {row['abattoir']} but no cattle prices for week {row['week']}! Please either fill in both or omit both, terminating!")
                if pd.isnull(row['value']):
                    warning(f"WARNING: There is a cattle price specified for cattle type {row['beef_type_cattle']} at abattoir {row['abattoir']} but no abattoir processing cost for week {row['week']}! Please either fill in both or omit both, terminating!")
            exit()

        df_processing_cost_data = df_processing_cost_data.dropna()
        df_processing_cost_data = df_processing_cost_data.drop(['beef_type', 'parameter', 'abattoir', 'beef_type_cattle', 'parameter_cattle', 'value_cattle', 'cattle', 'contract'], axis=1)

        # Now pivot back to a weekly format
        self.df_processing_cost = (
            pd.pivot_table(df_processing_cost_data,
                           values='value',
                           index='cattle_type',
                           columns='week',
                           aggfunc='sum')
                .reset_index()
        )

        # Report any missing values
        missing_processing_cost_rows = self.df_processing_cost[self.df_processing_cost.isnull().any(axis=1)]
        if not missing_processing_cost_rows.empty:
            for i, row in missing_processing_cost_rows.iterrows():
                parts = row['cattle_type'].split('_')
                steer = 'contract'
                if 'spot' in parts:
                    steer = 'spot'

                warning(
                    f"WARNING: Processing cost data not provided for all weeks for {parts[0]} {steer} steer at abattoir {parts[1]}, terminating!")
            exit()
        bqUploader('primary_processing_costs',self.df_processing_cost, self.scenario)
        # self.append_df_to_excel(self.workbook_loc, self.df_processing_cost, 'primary_processing_costs', 0, True)



    def process_steer_limits(self, df_combined_raw_abattoir_data):
        """
        Extracts the steer limits out of the raw abattoir data and writes to the input workbook.

        Args:
            df_combined_raw_abattoir_data (pd.DataFrame): The raw abattoir related data that required further processing
        """

        # Pull out the cattle cost df which has a nice list of the different cattle types that we wil reuse
        df_cattle_cost_data = self.cattle_cost_raw_data

        df_combined_raw_steer_limits = df_combined_raw_abattoir_data.loc[((df_combined_raw_abattoir_data['parameter'] == 'Committed') | (df_combined_raw_abattoir_data['parameter'] == 'Max available'))].copy()
        df_combined_raw_steer_limits = df_combined_raw_steer_limits.replace('Committed', 'min_steers')
        df_combined_raw_steer_limits = df_combined_raw_steer_limits.replace('Max available', 'max_steers')
        df_combined_raw_steer_limits['column_names'] = df_combined_raw_steer_limits['parameter'] + '_' + df_combined_raw_steer_limits['week']
        df_combined_raw_steer_limits = df_combined_raw_steer_limits.dropna()

        # Check all the limits make sense in terms of the max being larger or equal to the min
        df_combined_raw_steer_limits_check = df_combined_raw_steer_limits.copy()
        df_combined_raw_steer_limits_check.loc[df_combined_raw_steer_limits_check['parameter'] == 'min_steers', 'value'] = df_combined_raw_steer_limits_check.loc[df_combined_raw_steer_limits_check['parameter'] == 'min_steers', 'value'] * -1
        df_combined_raw_steer_limits_check_grouped = df_combined_raw_steer_limits_check.groupby(['beef_type', 'week', 'abattoir']).sum()
        df_combined_raw_steer_limits_check_grouped = df_combined_raw_steer_limits_check_grouped[df_combined_raw_steer_limits_check_grouped['value'] < 0]
        df_combined_raw_steer_limits_violated = df_combined_raw_steer_limits_check_grouped.loc[df_combined_raw_steer_limits_check_grouped['value'] < 0]
        if not df_combined_raw_steer_limits_violated.empty:
            for i, row in df_combined_raw_steer_limits_violated.iterrows():
                warning(f"WARNING: The maximum number of available steers is lower than the number of committed steers for steer type {i[0]} in week {i[1]} for abattoir {i[2]}. Number should include committed steers, terminating!")
            exit()

        # Also check that the minimum abattoir limit is not larger than the max steer limit and the min steer limit is not larger than the max abattoir limit
        df_combined_raw_steer_limits_check = df_combined_raw_steer_limits.copy()
        df_combined_raw_steer_limits_check = df_combined_raw_steer_limits_check.loc[df_combined_raw_steer_limits_check['beef_type'] != 'Total']
        df_combined_raw_steer_limits_total_min = df_combined_raw_steer_limits_check.loc[(df_combined_raw_steer_limits_check['parameter'] == 'min_steers')].groupby(['abattoir', 'week'], as_index=False).sum()
        df_combined_raw_steer_limits_total_max = df_combined_raw_steer_limits_check.loc[(df_combined_raw_steer_limits_check['parameter'] == 'max_steers')].groupby(['abattoir', 'week'], as_index=False).sum()
        df_combined_abattoir_limits_min = df_combined_raw_abattoir_data.loc[df_combined_raw_abattoir_data['parameter'] == 'Minimum total steers processed'].copy()
        df_combined_abattoir_limits_max = df_combined_raw_abattoir_data.loc[df_combined_raw_abattoir_data['parameter'] == 'Maximum total steers processed'].copy()
        df_combined_abattoir_limits_min = df_combined_abattoir_limits_min.merge(df_combined_raw_steer_limits_total_max, how='inner', on=['week', 'abattoir'], suffixes=['_abattoir', '_steer'])
        df_combined_abattoir_limits_max = df_combined_abattoir_limits_max.merge(df_combined_raw_steer_limits_total_min, how='inner', on=['week', 'abattoir'], suffixes=['_abattoir', '_steer'])

        df_combined_abattoir_limit_violations_min = df_combined_abattoir_limits_min.loc[df_combined_abattoir_limits_min['value_abattoir'] > df_combined_abattoir_limits_min['value_steer']]
        if not df_combined_abattoir_limit_violations_min.empty:
            for i, row in df_combined_abattoir_limit_violations_min.iterrows():
                warning(f"WARNING: The maximum number of available steers ({row['value_steer']}) is lower than the minimim number of steers ({row['value_abattoir']}) that abattoir {row['abattoir']} requires to be processed, terminating!")
            exit()

        df_combined_abattoir_limit_violations_max = df_combined_abattoir_limits_max.loc[df_combined_abattoir_limits_max['value_abattoir'] < df_combined_abattoir_limits_max['value_steer']]
        if not df_combined_abattoir_limit_violations_max.empty:
            for i, row in df_combined_abattoir_limit_violations_max.iterrows():
                warning(f"WARNING: The number of committed steers ({row['value_steer']}) is higher than the maximum number of steers ({row['value_abattoir']}) that abattoir {row['abattoir']} can process, terminating!")
            exit()

        df_combined_raw_steer_limits = df_combined_raw_steer_limits.merge(df_cattle_cost_data,
                                                                            how='outer',
                                                                            left_on=['abattoir', 'week', 'beef_type'],
                                                                            right_on=['abattoir', 'week', 'beef_type'],
                                                                            suffixes=[None, '_cattle'])

        # Check for missing data after merge and warn user of any gaps
        df_missing_data = df_combined_raw_steer_limits[df_combined_raw_steer_limits.isnull().any(axis=1)]
        df_missing_data = df_missing_data.loc[df_missing_data['beef_type'] != 'Total']
        if not df_missing_data.empty:
            for i, row in df_missing_data.iterrows():
                if pd.isnull(row['value_cattle']):
                    if row['parameter'] == 'min_steers':
                        warning(f"WARNING: There is a committed number of steer specified for cattle type {row['beef_type']} at abattoir {row['abattoir']} but no cattle price for week {row['week']}! Please either fill in both or omit both, terminating!")
                    else:
                        warning(f"WARNING: There is a maximum available number of steer specified for cattle type {row['beef_type']} at abattoir {row['abattoir']} but no cattle price for week {row['week']}! Please either fill in both or omit both, terminating!")

                if pd.isnull(row['value']):
                    if row['parameter'] == 'min_steers':
                        warning(f"WARNING: There is a cattle price specified for cattle type {row['beef_type']} at abattoir {row['abattoir']} but no committed steer number for week {row['week']}! Please either fill in both or omit both, terminating!")
                    else:
                        warning(f"WARNING: There is a cattle price specified for cattle type {row['beef_type']} at abattoir {row['abattoir']} but no maximum number of available steer for week {row['week']}! Please either fill in both or omit both, terminating!")
            exit()

        df_combined_raw_steer_limits = df_combined_raw_steer_limits.dropna()
        df_combined_raw_steer_limits = df_combined_raw_steer_limits.drop(['beef_type', 'parameter', 'abattoir', 'week', 'parameter', 'parameter_cattle', 'value_cattle', 'cattle', 'contract'], axis=1)

        # Now pivot back to a weekly format
        self.df_steer_limits = (
            pd.pivot_table(df_combined_raw_steer_limits,
                           values='value',
                           index='cattle_type',
                           columns='column_names',
                           aggfunc='sum')
                .reset_index()
        )

        # Report any missing values
        missing_limit_rows = self.df_steer_limits[self.df_steer_limits.isnull().any(axis=1)]
        if not missing_limit_rows.empty:
            for i, row in missing_limit_rows.iterrows():
                parts = row['cattle_type'].split('_')
                steer = 'contract'
                if 'spot' in parts:
                    steer = 'spot'

                warning(f"WARNING: Committed and maximum available steer numbers not provided for all weeks for {parts[0]} {steer} steer at abattoir {parts[1]}, terminating!")
            exit()
        bqUploader('steer_limits',self.df_steer_limits, self.scenario)

        # self.append_df_to_excel(self.workbook_loc, self.df_steer_limits, 'steer_limits', 0, True)



    def process_abattoir_limits(self, df_combined_raw_abattoir_data):
        """
        Extracts the abattoir limits out of the raw abattoir data and writes to the input workbook.

        Args:
            df_combined_raw_abattoir_data (pd.DataFrame): The raw abattoir related data that required further processing
        """

        df_combined_raw_abattoir_limits = df_combined_raw_abattoir_data.loc[((df_combined_raw_abattoir_data['parameter'] == 'Minimum total steers processed') | (df_combined_raw_abattoir_data['parameter'] == 'Maximum total steers processed'))].copy()
        df_combined_raw_abattoir_limits = df_combined_raw_abattoir_limits.replace('Minimum total steers processed', 'min_capacity')
        df_combined_raw_abattoir_limits = df_combined_raw_abattoir_limits.replace('Maximum total steers processed', 'max_capacity')
        df_combined_raw_abattoir_limits['column_names'] = df_combined_raw_abattoir_limits['parameter'] + '_' + df_combined_raw_abattoir_limits['week']

        # Check all the limits make sense in terms of the max being larger or equal to the min
        df_combined_raw_abattoir_limits_check = df_combined_raw_abattoir_limits.copy()
        df_combined_raw_abattoir_limits_check.loc[df_combined_raw_abattoir_limits_check['parameter'] == 'min_capacity', 'value'] = df_combined_raw_abattoir_limits_check.loc[df_combined_raw_abattoir_limits_check['parameter'] == 'min_capacity', 'value'] * -1
        df_combined_raw_abattoir_limits_grouped = df_combined_raw_abattoir_limits_check.groupby(['week', 'abattoir']).sum()
        df_combined_raw_abattoir_limits_grouped = df_combined_raw_abattoir_limits_grouped[df_combined_raw_abattoir_limits_grouped['value'] < 0]
        df_combined_raw_abattoir_limits_violated = df_combined_raw_abattoir_limits_grouped.loc[df_combined_raw_abattoir_limits_grouped['value'] < 0]
        if not df_combined_raw_abattoir_limits_violated.empty:
            for i, row in df_combined_raw_abattoir_limits_violated.iterrows():
                warning(f"WARNING: The maximum number of steers that can be processed is lower than the minimum number of steers that has to be processed in week {i[0]} for abattoir {i[1]}, terminating!")
            exit()

        df_combined_raw_abattoir_limits.drop(['beef_type', 'week', 'parameter'], axis=1, inplace=True)

        # Now pivot back to a weekly format
        self.df_abattoir_limits = (
            pd.pivot_table(df_combined_raw_abattoir_limits,
                           values='value',
                           index='abattoir',
                           columns='column_names',
                           aggfunc='sum')
                .reset_index()
        )
        bqUploader('abattoir_limits',self.df_abattoir_limits, self.scenario)

        # self.append_df_to_excel(self.workbook_loc, self.df_abattoir_limits, 'abattoir_limits', 0, True)



    def process_steer_weights(self, df_combined_raw_abattoir_data):
        """
        Extracts the steer weights out of the raw abattoir data and writes to the input workbook.

        Args:
            df_combined_raw_abattoir_data (pd.DataFrame): The raw abattoir related data that required further processing
        """
        df_combined_raw_steer_weights = df_combined_raw_abattoir_data.loc[(df_combined_raw_abattoir_data['parameter'] == 'Avg steer carcase weight')].copy()
        df_combined_raw_steer_weights = df_combined_raw_steer_weights.replace('Avg steer carcase weight', 'steer_weight')
        df_combined_raw_steer_weights['column_names'] = df_combined_raw_steer_weights['parameter'] + '_' + df_combined_raw_steer_weights['week']
        df_combined_raw_steer_weights = df_combined_raw_steer_weights.dropna()

        # Now pivot back to a weekly format
        df_steer_weights_limits = (
            pd.pivot_table(df_combined_raw_steer_weights,
                           values='value',
                           index=['abattoir', 'beef_type'],
                           columns='column_names',
                           aggfunc='sum')
                .reset_index()
        )

        # Merge onto the yield tree data we collected earlier and write into the input workbook
        self.df_steer_weights = self.df_yield_tree_data.merge(df_steer_weights_limits, how='outer', left_on=['abattoir', 'cattle_specification'], right_on=['abattoir', 'beef_type'])

        # Report any missing values
        missing_weight_rows = self.df_steer_weights[self.df_steer_weights.isnull().any(axis=1)]
        if not missing_weight_rows.empty:
            for i, row in missing_weight_rows.iterrows():
                if pd.isnull(row['yield_tree']):
                    warning(f"WARNING: No yield tree specified in the yield tree template for cattle type {row['cattle_specification']} at abattoir {row['abattoir']} but there are steer weights defined in the abattoir template, terminating!")
                elif pd.isnull(row['beef_type']):
                    warning(f"WARNING: No steer weights specified in the abattoir template for {row['cattle_specification']} cattle at abattoir {row['abattoir']} but there is a yield tree defined in the yield tree template, terminating!")
                else:
                    warning(f"WARNING: A steer weight for one of the weeks is missing in the abattoir template for {row['cattle_specification']} cattle at abattoir {row['abattoir']}, terminating!")
            exit()
        bqUploader('yield_trees',self.df_steer_weights, self.scenario)

        # self.append_df_to_excel(self.workbook_loc, self.df_steer_weights, 'yield_trees', 0, True)



    def read_in_abattoir_tab(self, file_path, sheet_name, abattoir):
        """
        Reads in a sheet from an abattoir template.

        Args:
              file_path (str): the path including the file name of the abattoir template excel workbook
              sheet_name (str): the name of the sheet to read in
              abattoir (str): the abattoir associated with the data sheet

        Returns:
              pd.DataFrame: a dataframe containing the raw abattoir related data found in the sheet
        """

        # Isolate the data we're interested in
        df_abattoir = pd.read_excel(file_path, sheet_name=sheet_name)
        df_abattoir.drop(['Unnamed: 0', 'Unnamed: 3'], axis=1, inplace=True)
        df_abattoir.columns = df_abattoir.iloc[4]
        if df_abattoir.iloc[4,0] != 'Beef Type':
            warning(f"WARNING: Layout of abattoir sheet {sheet_name} not as expected, please do not insert additional rows or columns above or before data, ignoring sheet!")
            return pd.DataFrame([])

        df_abattoir.drop(df_abattoir.index[0:5], inplace=True)
        df_abattoir = df_abattoir.dropna(how='all')
        if df_abattoir.empty:
            warning(f"WARNING: No data in abattoir sheet {sheet_name}!")
            return df_abattoir

        # Change the dimensions a bit
        df_abattoir = pd.melt(df_abattoir, id_vars=['Beef Type', 'Parameter'])
        df_abattoir.columns = ['beef_type', 'parameter', 'week', 'value']

        # Check if all the weeks are valid dates and convert to date part only
        try:
            df_abattoir['week'] = df_abattoir['week'].dt.date.astype(str)
        except AttributeError:
            warning(f"WARNING: The week date column headers in abattoir sheet {sheet_name} are not all valid dates, ignoring sheet!")
            return pd.DataFrame([])

        # Remove weeks we don't want
        weeks_to_keep = self.weeks
        df_abattoir = df_abattoir.loc[df_abattoir['week'].isin(weeks_to_keep)]
        if df_abattoir.empty:
            warning(f"WARNING: No data in abattoir sheet {sheet_name} for the weeks specified in the config file!")
        df_abattoir['abattoir'] = abattoir

        # Check if everything is a number
        try:
            df_abattoir['value'].astype(float)
        except ValueError:
            warning(f"WARNING: Not all values in abattoir sheet {sheet_name} are numbers, ignoring sheet!")
            return pd.DataFrame([])
        # print(df_abattoir[df_abattoir['beef_type']=='grsfed Supplementary'])
        return df_abattoir



    def load_suppl_primal_template(self):
        """
        Wrapper function to load in supplementary primal costs and limits
        """
        self.load_suppl_primal_costs()
        self.load_suppl_primal_limits()



    def load_suppl_primal_costs(self):
        """
        Loads the supplementary primal template to read in the costs of supplementary primals.
        """

        df_bb_suppl_costs = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'], 'BB_Costs', 'bb_MSA_', 'cost_per_kg')
        df_hw_suppl_costs = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'],'HW_Costs', 'hw_MSA_', 'cost_per_kg')
        df_trug_suppl_costs = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'],'TRUG_Costs', 'trug_MSA_', 'cost_per_kg')
        df_bb_GF_suppl_costs = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'], 'BB_GF_Costs', 'bb_grsfed_', 'cost_per_kg')
        df_bb_PR_suppl_costs = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'], 'BB_GF_Costs', 'bb_PR_', 'cost_per_kg')
        df_hw_GF_suppl_costs = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'], 'HW_GF_Costs', 'hw_grsfed_', 'cost_per_kg')
        df_hw_PR_suppl_costs = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'], 'HW_PR_Costs', 'hw_PR_', 'cost_per_kg')
        df_trug_GF_suppl_costs = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'], 'TRUG_GF_Costs', 'trug_grsfed_', 'cost_per_kg')
        df_trug_PR_suppl_costs = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'], 'TRUG_PR_Costs', 'trug_PR_', 'cost_per_kg')
        df_b2b_suppl_costs = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'], 'B2B_Costs', 'b2b_MSA_', 'cost_per_kg')

        df_raw_supl_primal_costs = df_bb_suppl_costs.append(df_hw_suppl_costs)
        df_raw_supl_primal_costs = df_raw_supl_primal_costs.append(df_trug_suppl_costs)
        df_raw_supl_primal_costs = df_raw_supl_primal_costs.append(df_bb_GF_suppl_costs)
        df_raw_supl_primal_costs = df_raw_supl_primal_costs.append(df_bb_PR_suppl_costs)
        df_raw_supl_primal_costs = df_raw_supl_primal_costs.append(df_hw_GF_suppl_costs)
        df_raw_supl_primal_costs = df_raw_supl_primal_costs.append(df_hw_PR_suppl_costs)
        df_raw_supl_primal_costs = df_raw_supl_primal_costs.append(df_trug_GF_suppl_costs)
        df_raw_supl_primal_costs = df_raw_supl_primal_costs.append(df_trug_PR_suppl_costs)
        df_raw_supl_primal_costs = df_raw_supl_primal_costs.append(df_b2b_suppl_costs)

        if df_raw_supl_primal_costs.empty:
            warning(f"WARNING: No supplementary primal costs found for any primals, we need at least some costs to optimise the supply, terminating!")
            exit()

        df_raw_supl_primal_costs.drop(['primal_description', 'week'], axis=1, inplace=True)

        # Now pivot back to a weekly format
        self.df_supl_primal_costs = (
            pd.pivot_table(df_raw_supl_primal_costs,
                           values='cost_per_kg',
                           index='primal_type',
                           columns='column_names',
                           aggfunc='sum')
                .reset_index()
        )
        bqUploader('supplementary_primal_costs',self.df_supl_primal_costs, self.scenario)

        # self.append_df_to_excel(self.workbook_loc, self.df_supl_primal_costs, 'supplementary_primal_costs', 0, True)



    def load_suppl_primal_limits(self):
        """
        Load the supplementary primal template to read in the limit on supplementary primals that might exist.
        """
        df_bb_suppl_limits = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'], 'BB_Availability', 'bb_MSA_', 'total_weight')
        df_hw_suppl_limits = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'],'HW_Availability', 'hw_MSA_', 'total_weight')
        df_trug_suppl_limits = self.read_in_suppl_primal_tab(self.file_dict['suppl_primal_template_loc'],'TRUG_Availability', 'trug_MSA_', 'total_weight')

        df_raw_supl_primal_limits = df_bb_suppl_limits.append(df_hw_suppl_limits)
        df_raw_supl_primal_limits = df_raw_supl_primal_limits.append(df_trug_suppl_limits)

        df_raw_supl_primal_limits.drop(['primal_description', 'week'], axis=1, inplace=True)

        # Now pivot back to a weekly format
        self.df_supl_primal_limits = (
            pd.pivot_table(df_raw_supl_primal_limits,
                           values='total_weight',
                           index='primal_type',
                           columns='column_names',
                           aggfunc='sum')
                .reset_index()
        )

        # It is possible that no limits have been specified at all. Create a dataframe with no entries
        if self.df_supl_primal_limits.empty:
            columns = ['primal_type']
            values = ['']
            for week in self.weeks:
                columns.append(('total_weight_' + week))
                values.append('')
            self.df_supl_primal_limits = pd.DataFrame(np.array([values]), columns=columns)
        bqUploader('supplemental_primal_limits',self.df_supl_primal_limits, self.scenario)

        # self.append_df_to_excel(self.workbook_loc, self.df_supl_primal_limits, 'supplemental_primal_limits', 0, True)



    def read_in_suppl_primal_tab(self, file_path, sheet_name, prefix, parameter):
        """
        Reads in a sheet from a supplementary primal template

        Args:
              file_path (str): the path including the file name of the suppl primal template excel workbook
              sheet_name (str): the name of the sheet to read in
              prefix (str): the prefix to add to the primal numbers for this crm
              parameter (str): the parameter we are collecting data for

        Returns:
              pd.DataFrame: a dataframe containing the raw suppl primal related data found in the sheet
        """

        # Isolate the data we're interested in
        df_prices = pd.read_excel(file_path, sheet_name=sheet_name)
        df_prices.drop(['Unnamed: 0', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        df_prices.columns = df_prices.iloc[4]
        if df_prices.iloc[4,0] != 'Product Number':
            warning(f"WARNING: Layout of supplementary primal sheet {sheet_name} not as expected, please do not insert additional rows or columns above or before data, ignoring sheet!")
            return pd.DataFrame([])
        df_prices.drop(df_prices.index[0:5], inplace=True)
        df_prices = df_prices.dropna(how='all')

        # Change the dimensions a bit
        df_prices = pd.melt(df_prices, id_vars=['Product Number', 'Product Name'])
        df_prices.columns = ['primal_type', 'primal_description', 'week', parameter]
        df_prices['primal_type'] = prefix + df_prices['primal_type'].astype(str)

        # Check if all the weeks are valid dates and convert to date part only
        try:
            df_prices['week'] = df_prices['week'].dt.date.astype(str)
        except AttributeError:
            warning(f"WARNING: The week date column headers in supplementary primal sheet {sheet_name} are not all valid dates, ignoring sheet!")
            return pd.DataFrame([])

        df_prices['column_names'] = parameter + '_' + df_prices['week']

        # Remove weeks we don't want and remove na entries
        weeks_to_keep = self.weeks
        df_prices = df_prices.loc[df_prices['week'].isin(weeks_to_keep)]
        df_prices = df_prices.dropna(how='any')

        # Check if everything is a number
        if not df_prices.empty:
            try:
                df_prices[parameter].astype(float)
            except ValueError:
                warning(f"WARNING: Not all values in sheet {sheet_name} are numbers, ignoring sheet!")
                return pd.DataFrame([])

        return df_prices



    def read_in_CRM_costs_tab(self, file_path: str, sheet_name: str, crm: str) -> pd.DataFrame:
        """
        Reads in a sheet from an allocation side costs template

        Args:
            file_path (str): the path including the file name of the retail costs template excel workbook
            sheet_name (str): the name of the sheet to read in
            crm (str): the CRM associated with the data sheet

        Returns:
            pd.DataFrame: a dataframe containing the crm cost data found in the sheet
        """

        # Isolate the data we're interested in
        df_CRM_costs = pd.read_excel(file_path, sheet_name=sheet_name)
        df_CRM_costs.drop(['Unnamed: 0', 'Unnamed: 3', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
                            'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12'], axis=1, inplace=True)
        df_CRM_costs.columns = df_CRM_costs.iloc[8]
        df_CRM_costs = clean_column_names(df_CRM_costs)
        df_CRM_costs.drop(df_CRM_costs.index[0:9], inplace=True)
        df_CRM_costs = df_CRM_costs.dropna(how='all')
        if len(df_CRM_costs) == 0:
            warning(f"WARNING: CRM costs tab {sheet_name} is empty.  Please fill it out.  If the tab IS NOT empty, check for trailing empty columns")
            exit()
        # convert everything that is numeric into a numeric, otherwise leave it as it is
        for col in ['yield_pc_target', 'raw_yield_pc_$_or_kg', 'proposed_sip_$_or_kg']:
            df_CRM_costs[col] = pd.to_numeric(df_CRM_costs[col], errors = 'ignore')
        df_CRM_costs['crm'] = crm
        df_CRM_costs = df_CRM_costs = df_CRM_costs.loc[:, df_CRM_costs.columns.notnull()]
        self.crm_costs_check(df_CRM_costs, sheet_name)

        return df_CRM_costs



    def crm_costs_check(self, df:pd.DataFrame, sheet_name:str):
        """
        Quick check for consistency of crm costs input

        Args:
            df (pd.DataFrame): the crm cost dataframe
            sheet_name (str): the name of the sheet
        """
        if list(df.columns) != ['description','wow_retail_no','yield_pc_target','raw_yield_pc_$_or_kg','proposed_sip_$_or_kg','crm']:
            warning(f"WARNING: CRM costs tab {sheet_name} is not of expected format.  Please do not add or remove columns to the template")
            exit()
        if not (pd.api.types.is_numeric_dtype(df['yield_pc_target']) 
                and pd.api.types.is_numeric_dtype(df['raw_yield_pc_$_or_kg'])
                and pd.api.types.is_numeric_dtype(df['proposed_sip_$_or_kg'])):
            warning(f"WARNING: CRM costs tab {sheet_name} contains non-numeric data in the required numeric columns.  Please fix these columns and try again")
            exit()



    def _load_abattoir_packing_costs(self, cattle_type:str):
        """
        Calculate the packing costs for a given channel

        Args:
            cattle_type (str): type of cattle to load.  currently must be 'msa' or 'hgf'
        """
        if (cattle_type != "msa") and (cattle_type != "hgf"):
            warning("WARNING: cattle_type input not valid.  Please choose 'msa' or 'hgf'")
            exit()
        # packaging costs for retail
        df_abattoir_packing = (
            clean_column_names(pd.read_excel(self.file_dict[f'abattoir_packaging_{cattle_type}']))
                .query('item_type != "Cost"')
                .rename(columns = {'abattoir_cost':'packaging_cost_per_kg'})
        )
        if len(df_abattoir_packing) == 0:
            warning(f"WARNING: Foods Connected abattoir cost sheet is empty.  Please check the recheck the input sheet.")
            exit()
        df_abattoir_packing = (
            df_abattoir_packing[~df_abattoir_packing['packaging_cost_per_kg'].isna()]
            .rename(columns = {'item_code':'primal_article'})
            .merge(self.df_cost_group_map.rename(columns = {'product_code':'primal_article'}),
                                                how = 'left',
                                                on = 'primal_article')
        )
        df_abattoir_packing = df_abattoir_packing[~df_abattoir_packing['primal_article'].astype('str').str.contains('-')]
        abattoir_packing_list = ([df_abattoir_packing.assign(primal_name =
                                        lambda df: cattle_spec + df['primal_article'].astype(str)) for cattle_spec in self.cattle_types if cattle_type.upper() in cattle_spec])
        df_abattoir_packing = pd.concat(abattoir_packing_list)[['primal_name','packaging_cost_per_kg']]
        if not pd.api.types.is_numeric_dtype(df_abattoir_packing['packaging_cost_per_kg']):
            warning(f"WARNING: Packaging costs are not numeric.  Please check and correct the data.")
            exit()

        return df_abattoir_packing


    def pivot_into_weeks(self, df:pd.DataFrame, initial_cols:StringList, converted_cols:StringList, index_col:str):
        """
        Pivot a dataframe with a number of columns and a weeks column into columns based on the intiial columns and the weeks.

        Args:
            df (pd.DataFrame): The dataframe to transform
            initial_cols (StringList): the names of the initial columns to transform
            converted_cols (StringList): prefix for the converted columns
            index_col (str): the index column

        Returns:
            pd.DataFrame: a dataframe that has pivoted the inputs based on the number of weeks
        """
        # first need the check that the length of initial_cols and coverted_cols are the same as we basically loop through them
        if len(initial_cols) != len(converted_cols):
            warning("WARNING: Attempting to transform a dataframe without a matching number of initial and final columns")
            exit()
        if ('week' not in list(df.columns)):
            warning("WARNING: Input dataframe does not contain a 'week' column")
            exit()
        
        # now generate a list of dataframes
        df_list = []
        for i in range(len(initial_cols)):
            df_list.append(pd.pivot_table(df, values = initial_cols[i], index = index_col, columns = 'week').reset_index())
            # print(list(df_list[i].columns))
            df_list[i].columns = [converted_cols[i] + date if date != index_col else date for date in list(df_list[i].columns.values)]
        
        df_output = df_list[0]
        for i in range(1,len(df_list)):
            df_output = pd.merge(df_output,
                                    df_list[i],
                                    on = index_col,
                                    how = 'outer')
        
        return df_output
        


    def load_export_sales(self) -> pd.DataFrame:
        """
        Loads in the export sales data template and converts to primal sales information

        Returns:
            pd.DataFrame: A dataframe containing median export sales prices for hgf cattle
        """

        df_msa_export = self.read_in_export_price_tab(self.file_dict['export_msa_demand_loc'], 'Export Customer Demand Pricing','hw_MSA_')
        df_hgf_export = self.read_in_export_price_tab(self.file_dict['export_hgf_demand_loc'], 'Hybrid Export Pricing','HGF_')
        df_export = pd.concat((df_msa_export, df_hgf_export))
        # this is a bit empty now, but is a placeholder for when retail and b2b information on hgf cattle comes through
        return df_export



    def read_in_export_price_tab(self, file_path: str, sheet_name: str, prefix: str) -> pd.DataFrame:
        """
        Reads in a price sheet from the export demand file and converts it to necessary

        Args:
            file_path (str): the path including the file name of the demand template excel workbook
            sheet_name (str): the name of the sheet to read in
            prefix (str): the prefix to attach to the primal codes for this demand

        Returns:
            pd.DataFrame: a dataframe containing the pricing by primal per week
        """

        # Isolate the data we're interested in
        df_export_prices = pd.read_excel(file_path, sheet_name=sheet_name)
        df_export_prices.drop(['Unnamed: 0', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        if df_export_prices.iloc[8,0] != 'Product Number':
            warning("WARNING: The format of the export price template has been changed.  Please do not add and rows or columns to the template")
            exit()
        df_export_prices.columns = df_export_prices.iloc[8]
        df_export_prices.drop(df_export_prices.index[0:9], inplace=True)
        df_export_prices = df_export_prices.dropna()
        if len(df_export_prices) <= 0:
            warning(f"WARNING: Empty export prices sheet at {sheet_name}.  If this sheet is not empty, there may be some trailing empty columns")
            exit()

        # Change the dimensions a bit
        df_export_prices = pd.melt(df_export_prices, id_vars=['Product Number'])
        df_export_prices.columns = ['primal', 'week', 'export_price']
        df_export_prices['week'] = df_export_prices['week'].dt.date.astype(str)
        df_export_prices['primal'] = prefix + df_export_prices['primal'].astype(str)
        df_export_prices['export_price'] = pd.to_numeric(df_export_prices['export_price'])

        # Remove weeks we don't want
        df_export_prices = df_export_prices.loc[df_export_prices['week'].isin(self.weeks)]
        if len(df_export_prices) <= 0:
            warning(f"WARNING: Weeks don't match with config weeks")
            exit()

        # Save prefix for later
        if prefix not in self.cattle_types:
            self.cattle_types.append(prefix)    

        return df_export_prices


    
    def load_msa_sales(self, use_asp:bool) -> pd.DataFrame:
        """
        Loads in the retail pricing data and converts to primal pricing information

        Args:
            use_asp (bool): use ASP values from SIP data to determine retail sales pricing

        Returns:
            pd.DataFrame: A dataframe containing median retail sales prices for groupings of primals
        """

        if use_asp:
            df_trug_retail = self.read_in_ASP_tab(self.file_dict['crm_costs_template_loc'], 'TRUG_SIP_Items', 'trug_MSA_')
            df_bb_retail = self.read_in_ASP_tab(self.file_dict['crm_costs_template_loc'], 'BB_SIP_Items', 'bb_MSA_')
            df_hw_retail = self.read_in_ASP_tab(self.file_dict['crm_costs_template_loc'], 'HW_SIP_Items', 'hw_MSA_')
            df_hw_retail.loc[df_hw_retail['description'].str.contains('Grassfed', na=False),'primal'] = df_hw_retail.loc[
                df_hw_retail['description'].str.contains('Grassfed', na=False),'primal'].replace(
                'hw_MSA', 'hw_grsfed', regex=True)

            df_hw_retail.loc[df_hw_retail['description'].str.contains('PR', na=False), 'primal'] = \
            df_hw_retail.loc[
                df_hw_retail['description'].str.contains('PR', na=False), 'primal'].replace(
                    'hw_MSA', 'hw_PR', regex=True)

            df_retail = (
                pd.concat([df_trug_retail, df_bb_retail, df_hw_retail])
                    .groupby(['primal','week'])
                    .agg(retail_price = ('asp','median'))
                    .reset_index()
            )
        else:
            # load up WA sales price
            df_wa_retail = [self.read_in_retail_price_tab(self.file_dict['rrp_template_loc'], 'WA_Retail_Items', 'bb_MSA_')]
            # load up eastern seaboard sales prices (note that this includes NT and SA)
            df_ESB_retail = [self.read_in_retail_price_tab(self.file_dict['rrp_template_loc'], state, cattle_type) 
                                for state in self.esb_state_list 
                                for cattle_type in ['trug_MSA_','hw_MSA_']]
            # determine the median of the retail price (as primals will come from multiple plants to ESB)
            df_retail = (
                pd.concat(df_wa_retail + df_ESB_retail)
                    .assign(retail_price = lambda df: pd.to_numeric(df['retail_price']))
                    .groupby(['primal','week'])
                    .agg(retail_price = ('retail_price','median'))
                    .reset_index()
            )



        # using historical AFG prices now
        df_export = self.read_in_historical_AFG_sales(self.file_dict['afg_historical_loc'])

        # 
        df_b2b = (
            self.read_in_retail_price_tab(self.file_dict['b2b_price_template_loc'], 'B2B_Prices', 'b2b_MSA_', convert_sku_to_primal = False)
                .assign(b2b_price = lambda df: pd.to_numeric(df['retail_price']))
                .drop('retail_price', axis = 1)
        )


        df_msa_prices = (df_retail.merge(df_export,
                                        how = 'outer',
                                        on = ['primal','week'])
                                    .merge(df_b2b,
                                            how = 'outer',
                                            on = ['primal','week'])
        )

        # imputing missing AFG export prices at 50% of the retail prices
        # info("Imputing missing AFG prices for MSA overflow using 50% of retail sales value")
        # df_msa_prices = df_msa_prices.assign(missing_prices = lambda df: df['retail_price']*0.5)
        # df_msa_prices.loc[df_msa_prices['export_price'].isna(),'export_price'] = df_msa_prices.loc[df_msa_prices['export_price'].isna(),'missing_prices']
        # df_msa_prices.drop('missing_prices', axis = 1, inplace = True)

        #impute missing b2b prices with 0 for thisself.scenariorun
        # df_msa_prices['b2b_price'].fillna(0, inplace = True)


        return df_msa_prices



    def read_in_ASP_tab(self, file_path:str, sheet_name:str, prefix: str) -> pd.DataFrame:
        """
        Reads in the ASP prices that are attached to SIP data

        Args:
            file_path (str): location of SIP data
            sheet_name (str): sheet name to read in
            prefix (str): primal name prefix to add to the primal list

        Returns:
            pd.DataFrame: recent historical retail prices for SKUs in ASP
        """

        df_asp = pd.read_excel(file_path, sheet_name = sheet_name)
        df_asp.drop([ 'Unnamed: 3',  'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7',
                        'Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12'], axis=1, inplace=True)
        if df_asp.iloc[8,2] != 'WOW Retail #':
            warning("WARNING: The format of the retail price template has been changed.  Please do not add and rows or columns to the template")
            exit()
        # print(df_asp.iloc[8])
        # print(df_asp.columns)
        df_asp.columns = df_asp.iloc[8]
        df_asp.drop(df_asp.index[0:9], inplace=True)

        df_asp = df_asp.iloc[:, : 5]
        
        df_asp = df_asp.dropna()

        df_asp = clean_column_names(df_asp)

        if len(df_asp) <= 0:
            warning(f"WARNING: Empty ASP column on {sheet_name}.  Please fill this column")
            exit()
        # print(df_asp.columns)
        df_asp = (
            pd.concat([df_asp.assign(week = week) for week in self.weeks])
                .merge(self.df_sku_to_primal.rename(columns = {'primal_article':'primal'}),
                        how = 'left',
                        on = 'wow_retail_no')
                .assign(primal = lambda df: prefix + df['primal'],
                        asp = lambda df: pd.to_numeric(df['asp']) * pd.to_numeric(df['yield_pc_target']))
                .drop(['wow_retail_no','yield_pc_target'], axis = 1)
        )

        if len(df_asp[df_asp['primal'].isna()]) > 0:
            warning(f"WARNING: SKU to primal mapping missing at maximum {len(df_asp[df_asp['primal'].isna()])} SKUs.  This number may be inflated by the number of weeks.  Filtering out missing SKUs...")
            df_asp = df_asp[~df_asp['primal'].isna()]

        if len(df_asp) <= 0:
            warning(f"WARNING: Weeks don't match with config weeks")
            exit()

        # print(df_asp)

        return df_asp

        

    def read_in_historical_AFG_sales(self, file_path: str) -> pd.DataFrame:
        """
        Reads in historical AFG sales to use for overflow retail price points in export

        Args:
            file_path (str): location of historical AFG sales data

        Returns:
            pd.DataFrame: results of AFG sales data in a form
        """

        df_afg_prices = (
            clean_column_names(pd.read_excel(file_path))[['item_code','price_confirmed_ex_works_$']]
                .rename(columns = {'item_code':'primal_number', 'price_confirmed_ex_works_$':'export_price'})
                .dropna()
                .groupby('primal_number')
                .agg(export_price = ('export_price','mean'))
                .reset_index()
        )
        df_afg_prices['primal_number'] = df_afg_prices['primal_number'].astype(int).astype(str)
        df_afg_prices = [df_afg_prices.assign(primal = lambda df: cattle_spec + df['primal_number'], 
                                                week = week)
                            for cattle_spec in ['bb_MSA_','hw_MSA_','trug_MSA_']
                            for week in self.weeks]
        df_afg_prices = pd.concat(df_afg_prices)[['primal','week','export_price']]

        return df_afg_prices



    def read_in_retail_price_tab(self, file_path: str, sheet_name: str, prefix: str, convert_sku_to_primal:bool = True) -> pd.DataFrame:
        """
        Reads in a sheet from a retail price template

        Args:
            file_path (str): the path including the file name of the demand template excel workbook
            sheet_name (str): the name of the sheet to read in
            prefix (str): the prefix to attach to the primal codes for this demand
            demand_type (str): the type of demand - retail/export/b2b
            convert_sku_to_primal (bool): If given a list of skus convert these to primals.  Usual use: set to False if loading b2b primal sales prices

        Returns:
            pd.DataFrame: a dataframe containing the demand by primal and week
        """

        # Isolate the data we're interested in
        df_rrp = pd.read_excel(file_path, sheet_name=sheet_name)
        df_rrp.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1, inplace=True)
        if df_rrp.iloc[8,0] != 'Article':
            warning("WARNING: The format of the retail price template has been changed.  Please do not add and rows or columns to the template")
            exit()
        df_rrp.columns = df_rrp.iloc[8]
        df_rrp.drop(df_rrp.index[0:9], inplace=True)
        df_rrp = df_rrp.dropna()
        if len(df_rrp) <= 0:
            warning(f"WARNING: Empty export prices sheet at {sheet_name}.  If this sheet is not empty, there may be some trailing empty columns")
            exit()

        # Change the dimensions a bit
        df_rrp = pd.melt(df_rrp, id_vars=['Article'])
        df_rrp.columns = ['Article', 'week', 'retail_price']

        if convert_sku_to_primal:
            df_rrp = (
                df_rrp.merge(self.df_sku_to_primal.rename(columns = {'primal_article':'primal'}),
                                how = 'left',
                                on = 'wow_retail_no')
                    .drop('wow_retail_no', axis = 1)
            )

            if len(df_rrp[df_rrp['primal'].isna()]) > 0:
                warning(f"WARNING: SKU to primal mapping missing at maximum {len(df_rrp[df_rrp['primal'].isna()])} SKUs.  This number may be inflated by the number of weeks.  Filtering out missing SKUs...")
                df_rrp = df_rrp[~df_rrp['primal'].isna()]

            df_rrp['week'] = df_rrp['week'].dt.date.astype(str)
            df_rrp['primal'] = prefix + df_rrp['primal'].astype(str)
            if len(df_rrp) <= 0:
                warning(f"WARNING: Weeks don't match with config weeks")
                exit()
        else:
            df_rrp = (df_rrp
                        .rename(columns = {'Article':'primal'})
                        .assign(primal = lambda df: prefix + df['primal'].astype(str))
            )
            try:
                #if the week is read in a datatime objects, convert to string
                df_rrp['week'] = df_rrp['week'].dt.strftime("%Y-%m-%d")
            except:
                pass

        # Remove weeks we don't want
        weeks_to_keep = self.weeks
        df_rrp = df_rrp.loc[df_rrp['week'].isin(weeks_to_keep)]
        # Save prefix for later
        if prefix not in self.cattle_types:
            self.cattle_types.append(prefix)    


        return df_rrp


    def read_in_retail_tab(self, file_path, sheet_name, channel):
        """
                Reads in a sheet from a demand template.

                Args:
                      sheet_name (str): the name of the sheet to read in
                      channel (str): the channel in which the demand is attributed to. Can be 'retail', 'b2b', or 'export'


                Returns:
                      pd.DataFrame: a dataframe containing the demand by primal and week
                """

        # Isolate the data we're interested in
        df_demand = pd.read_excel(file_path, sheet_name=sheet_name)

        df_demand['Channel'] = channel

        # Replace clean up volumes and convert to integers
        vol_col = ['Primal kgs for production', 'Proposed Purchases', 'Confirmed Purchases', 'Low Codes',
                   'Op Stock', 'Unmature stock', 'Butcher shop Rq.']

        for each in vol_col:
            df_demand[each].replace(',', '')
            df_demand[each].fillna(0, inplace=True)
            df_demand[each].astype(int)

        return df_demand


    def read_in_channel_demand(self, file_path, sheet_name, channel):
        """
                Reads in a sheet from a demand template.

                Args:
                      file_path (str): the file path to the demand template
                      sheet_name (str): the name of the sheet to read in
                      channel (str): the channel in which the demand is attributed to. Can be 'retail', 'b2b', or 'export'


                Returns:
                      pd.DataFrame: a dataframe containing the demand by primal and week
                """

        df_demand = pd.read_excel(file_path, sheet_name=sheet_name)

        df_demand.columns = df_demand.iloc[8]
        df_demand = df_demand.iloc[9:, 1:]
        df_demand.reset_index(drop=True, inplace=True)
        df_demand = df_demand.melt(id_vars=['Product Number', 'Product Name', 'Parameter', 'Units'])
        df_demand = df_demand.rename(columns={8: 'Date', 'value': 'Proposed Purchases', 'Product Number': 'PrimalID',
                                              'Product Name': 'PrimalName'})
        df_demand = df_demand.assign(**{'Primal kgs for production': 0, 'Unmature stock': 0,
                                        'Butcher shop Rq.': 0, 'Confirmed Purchases': 0, 'Family': 'BEEF', 'Plant': 'M003',
                                        'Low Codes': 0, 'Op Stock': 0, 'Channel': channel})
        df_demand['WOW code'] = df_demand['PrimalID']
        df_demand = df_demand.drop(columns=['Parameter', 'Units'])
        df_demand = df_demand[df_demand['Proposed Purchases'].notnull()]

        return df_demand

    def read_in_vendor_demand(self, file_path, sheet_name, channel):
        """
                Reads in a sheet from a demand template.

                Args:
                      file_path (str): the file path to the demand template
                      sheet_name (str): the name of the sheet to read in
                      channel (str): the channel in which the demand is attributed to. Can be 'retail', 'b2b', or 'export'


                Returns:
                      pd.DataFrame: a dataframe containing the demand by primal and week
                """

        # Isolate the data we're interested in
        df_demand = pd.read_excel(file_path, sheet_name=sheet_name)

        # Add in missing columns
        df_demand = df_demand.assign(**{'Channel': channel, 'Primal kgs for production': 0, 'Unmature stock': 0,
                                        'Butcher shop Rq.': 0})
        df_demand = df_demand.drop(columns=['Sales Demand'])

        # Replace clean up volumes and convert to integers
        vol_col = ['Primal kgs for production', 'Proposed Purchases', 'Confirmed Purchases', 'Low Codes',
                   'Op Stock', 'Unmature stock', 'Butcher shop Rq.']

        for each in vol_col:
            df_demand[each].replace(',', '')
            df_demand[each].fillna(0, inplace=True)
            df_demand[each].astype(int)

        return df_demand

    def check_col_names(self, dem_date):
        """

        :type dem_date: str
        """
        retail_col = 'weight_' + 'retail_' + dem_date
        wholesale_col = 'weight_' + 'b2b_' + dem_date
        international_col = 'weight_' + 'export_' + dem_date

        return [retail_col, wholesale_col, international_col]


    def load_demand(self):

        species = ['BEEF']

        raw_demand_hw = self.read_in_retail_tab(self.file_dict["retail_demand_vol_loc"], sheet_name='HW', channel='retail')
        raw_demand_trug = self.read_in_retail_tab(self.file_dict["retail_demand_vol_loc"], sheet_name='TRUG', channel='retail')
        raw_demand_bb = self.read_in_retail_tab(self.file_dict["retail_demand_vol_loc"], sheet_name='BUN', channel='retail')
        raw_vendor_hw = self.read_in_vendor_demand(self.file_dict["vendorline_demand_vol_loc"], sheet_name='HW', channel='retail')
        raw_vendor_trug = self.read_in_vendor_demand(self.file_dict["vendorline_demand_vol_loc"], sheet_name='TRUG', channel='retail')
        raw_vendor_bb = self.read_in_vendor_demand(self.file_dict["vendorline_demand_vol_loc"], sheet_name='BUN', channel='retail')
        raw_demand_wholesale = self.read_in_channel_demand(self.file_dict["wholesale_demand_vol_loc"], sheet_name='B2B_Volumes', channel='b2b')
        raw_demand_export = self.read_in_channel_demand(self.file_dict["export_customer_demand_vol_loc"], sheet_name='Export Customer Demand Volumes', channel='export')
        raw_demand_hybrid = self.read_in_channel_demand(self.file_dict["hybrid_customer_demand_vol_loc"], sheet_name='Hybrid Export Volumes', channel='export')


        raw_demand_df = pd.concat([raw_demand_hw, raw_demand_trug, raw_demand_bb,
                                   raw_vendor_hw, raw_vendor_trug, raw_vendor_bb,
                                   raw_demand_wholesale, raw_demand_export, raw_demand_hybrid])
        raw_demand_df.reset_index(drop=True, inplace=True)

        raw_demand_df.loc[raw_demand_df['Plant'] == 'M001', 'CRM Site'] = 'bb'
        raw_demand_df.loc[raw_demand_df['Plant'] == 'M002', 'CRM Site'] = 'trug'
        raw_demand_df.loc[raw_demand_df['Plant'] == 'M003', 'CRM Site'] = 'hw'

        # Date to datetime format
        raw_demand_df['Date'] = pd.to_datetime(raw_demand_df['Date'], format='%d/%m/%Y').dt.date
        # Creates a column with the week commencing Monday
        raw_demand_df['Week Commencing'] = raw_demand_df['Date'].apply(
            lambda date: date - timedelta(days=date.weekday()))

        # Outlines beef type
        raw_demand_df.loc[raw_demand_df['PrimalName'].str.contains('GRASS', na=False), 'Type'] = 'grsfed'
        raw_demand_df.loc[raw_demand_df['PrimalName'].str.contains('PR\s', na=False), 'Type'] = 'PR'
        raw_demand_df.loc[raw_demand_df['PrimalName'].str.contains('100[GDd]|100\s', na=False), 'Type'] = 'HGF'
        raw_demand_df.loc[raw_demand_df['Type'].isnull(), 'Type'] = 'MSA'

        # Clean up wow codes in preparation for concatenation

        # Convert Column data to strings and concatenate
        raw_demand_df['WOW code'] = raw_demand_df['WOW code'].astype(str)
        raw_demand_df['Week Commencing'] = raw_demand_df['Week Commencing'].astype(str)
        raw_demand_df['Proposed Purchases'] = raw_demand_df['Proposed Purchases'].astype(int)
        raw_demand_df['primal'] = raw_demand_df['CRM Site'] + '_' + raw_demand_df['Type'] + '_' + raw_demand_df['WOW code']
        raw_demand_df['Channel Weight'] = 'weight_' + raw_demand_df['Channel'] + '_' + raw_demand_df['Week Commencing']


        # Filter rows based on criteria
        updated_df = raw_demand_df[raw_demand_df['Week Commencing'].isin(self.weeks)]
        updated_df = updated_df.loc[updated_df['Family'].isin(species)]

        # Drop unnecessary columns
        updated_df = updated_df.drop(columns=['Date', 'PrimalID', 'Family', 'PrimalName',
                                              'Primal kgs for production', 'Confirmed Purchases',
                                              'Low Codes', 'Op Stock', 'Unmature stock', 'Butcher shop Rq.',
                                              'WOW code', 'Plant', 'Channel', 'CRM Site', 'Week Commencing', 'Type'])


        updated_df = pd.pivot_table(updated_df, values='Proposed Purchases', index='primal', columns='Channel Weight', aggfunc=np.sum)
        updated_df.reset_index(inplace=True)

        # check if all channel columns for given dates are present if not add column to dataframe


        col_names = [self.check_col_names(dem_date) for dem_date in self.weeks]
        updated_col_names = [item for sublist in col_names for item in sublist]

        for each in updated_col_names:
            if each not in list(updated_df.columns):
                updated_df[each] = 0
            continue

        # replace nan values with zeros and re-sorts columns
        self.df_updated_demand = updated_df.fillna(0)

        bqUploader('demand',self.df_updated_demand, self.scenario)