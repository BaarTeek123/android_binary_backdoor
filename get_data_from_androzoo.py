import pandas as pd
import os
from config import key
import sys
from lxml import etree


malware = pd.read_csv('malware.csv')['sha256'][:5]


def check_permissions_in_manifest(file_path, permissions):
    try:
        return _extracted_from_check_permissions_in_manifest_(file_path, permissions)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML file: {file_path}")
        sys.exit(1)



def _extracted_from_check_permissions_in_manifest_(file_path, permissions):
    parser = etree.XMLParser(ns_clean=True, recover=True)
    tree = etree.parse(file_path, parser=parser)
    root = tree.getroot()

    permission_elements = root.xpath("./uses-permission")

    existing_permissions = set()
    for elem in permission_elements:
        permission_name = elem.get("{http://schemas.android.com/apk/res/android}name")
        if permission_name is not None:
            existing_permissions.add(permission_name)

    return [perm in existing_permissions for perm in permissions]


def create_or_update_csv(file_name, values, columns, index_variable):
    # Create a new DataFrame from the given values, columns, and index
    df = pd.DataFrame(values, columns=columns, index=index_variable)
    
    # Check if the file exists
    if os.path.isfile(file_name):
        # If the file exists, read it and update it with new data
        existing_df = pd.read_csv(file_name, index_col=0)
        updated_df = existing_df.append(df)
    else:
        # If the file does not exist, use the new DataFrame as is
        updated_df = df

    # Save the updated DataFrame to the CSV file
    updated_df.to_csv(file_name)


permissions_to_check = [
    "android.permission.INTERNET",
    "android.permission.CAMERA",
    "android.permission.ACCESS_FINE_LOCATION",
]  


# output_path = 'output_non_malware.csv'
output_path = 'output_malware.csv'
for sha256 in malware:
    print(sha256)
    os.system(f'curl -O --remote-header-name -G -d apikey={key} -d sha256={sha256} https://androzoo.uni.lu/api/download')
    os.system(f'apktool d {sha256}.apk')
    create_or_update_csv(output_path, [check_permissions_in_manifest(f'./{sha256}/AndroidManifest.xml', permissions_to_check)], columns=permissions_to_check, index_variable={sha256})
    os.system(f'rm -rf {sha256}.apk')
    os.system(f'rm -rf {sha256}')

# curl -O --remote-header-name -G -d apikey=71d13696584402f5b5cf5b9daef42d1d8a11379eb33b4a9f1952f0e50d244f19 -d sha256=1E3CD4BD200D8CF71985560A3B206E5E8A3DA664F68A0D1D05BFFE47F8269429 https://androzoo.uni.lu/api/download


# file = "./data/000027D1DA96332EFCB54AF76906A7298121EBCCCDAB3D7DCE999F8043E74EE7/AndroidManifest.xml"

# print(check_permissions_in_manifest(file, permissions_to_check))

