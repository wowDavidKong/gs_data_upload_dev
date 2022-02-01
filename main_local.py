import sys
sys.path.insert(1, '..')
from src.input_workbook_preparation import WorkbookPrep
from flask import jsonify


def uploader():
    # Set CORS headers for the preflight request
    
    scenarioName = "test_123"
    weeks = ['2021-12-06']

    fileDict = {}
    fileDict["retail_demand_vol_loc"] = 'C:/Users/1162723/gs_optimiser_data_uploader_dev/Test Files/retail_demand_vol.xlsx'
    fileDict["vendorline_demand_vol_loc"] = 'C:/Users/1162723/gs_optimiser_data_uploader_dev/Test Files/vendorline_demand_vol.xlsx'
    fileDict["wholesale_demand_vol_loc"] = 'C:/Users/1162723/gs_optimiser_data_uploader_dev/Test Files/wholesale_demand_vol.xlsx'
    fileDict["export_customer_demand_vol_loc"] ='C:/Users/1162723/gs_optimiser_data_uploader_dev/Test Files/export_customer_demand_vol.xlsx'
    fileDict["hybrid_customer_demand_vol_loc"] ='C:/Users/1162723/gs_optimiser_data_uploader_dev/Test Files/hybrid_customer_demand_vol.xlsx'
    fileDict["hilton_yield_data"] = "C:/Users/1162723/gs_optimiser_data_uploader_dev/Test Files/hilton_yield_data.xlsx"

    new_workbook = WorkbookPrep(fileDict,weeks,scenarioName)
    if new_workbook.scenario_manager():
        new_workbook.load_demand()
        # new_workbook.load_yield_tree_template()
        # new_workbook.load_abattoir_template()
        # new_workbook.load_suppl_primal_template()
        # new_workbook.load_primal_sales_prices()
        # Run prices before costs to get the percentage based AFG costs out
        # new_workbook.load_allocation_side_costs()
        

uploader()