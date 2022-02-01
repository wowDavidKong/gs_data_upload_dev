import sys
sys.path.insert(1, '..')
from src.input_workbook_preparation import WorkbookPrep
from flask import jsonify


def uploader(request):
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    print("FUNCTION STARTING")
   
 
    fileDict = {}
    f = request.form
    scenarioName = f["scenarioname"]
    weeks = f["weeks"]
    weeks = weeks.split(',')
    d = request.files

    print(weeks)
    print(scenarioName)
    
    for key in d:
        fileName = key
        xlFile = d[key]
        fileDict[fileName] = xlFile

    new_workbook = WorkbookPrep(fileDict,weeks,scenarioName)
    if new_workbook.scenario_manager():
        try:
            new_workbook.load_demand()
        except:
            return (jsonify({"status":"Failed at loading demand templates. Please check excel files and check stackdriver logs for more details."}),200,headers)
        try: 
            new_workbook.load_yield_tree_template()
        except: 
            return (jsonify({"status":"Failed at loading yield tree templates. Please check excel files and check stackdriver logs for more details."}),200,headers)
        try:
            new_workbook.load_abattoir_template()
        except: 
            return (jsonify({"status":"Failed at loading abattoir templates. Please check excel files and check stackdriver logs for more details."}),200,headers)
        try: 
            new_workbook.load_suppl_primal_template()
        except: 
            return (jsonify({"status":"Failed at loading supp primal templates. Please check excel files and check stackdriver logs for more details."}),200,headers)
        # new_workbook.load_primal_sales_prices()
        # Run prices before costs to get the percentage based AFG costs out
        # new_workbook.load_allocation_side_costs()
        return (jsonify({"status":"success"}),200,headers)
