from itertools import chain
import xml.etree.ElementTree as ET
import pandas as pd
from os import listdir


def extract_permissions(manifest_file_path):
    with open(manifest_file_path, "r", encoding="utf-8-sig") as manifest:
        root = ET.parse(manifest).getroot()
        return list(chain.from_iterable([list(k.attrib.values()) for k in root.findall("uses-permission")]))


def check_permissions_in_manifest(file_path, permissions: list, just_bools=True):
    try:
        if permissions is not None and just_bools:
            return [any(item.lower().endswith(permission.lower()) for item in list(extract_permissions(file_path))) for
                    permission in permissions]
            # return [any(permission.lower() in item.lower() for item in list(extract_permissions(file_path))) for permission in permissions]
        elif permissions is not None:
            return extract_permissions(file_path)

        # return _extracted_from_check_permissions_in_manifest_(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        # sys.exit(1)


#     except etree.XMLSyntaxError as e:
#         print(f"Error parsing XML file: {file_path} ")


# function to merge info check permissions in Android manifest file (in a directory -> in our case a directory with malware and non-malware files)
def create_merged_from_manifest(manifest_file_dir: str, permissions_list: list, file_name: str = None):
    merged_df = pd.DataFrame(columns=permissions_list)
    for sha256 in listdir(manifest_file_dir):
        try:
            df = pd.DataFrame([check_permissions_in_manifest(f'{manifest_file_dir}/{sha256}', permissions_list)],
                              columns=permissions_list, index=[sha256])
            merged_df = pd.concat([merged_df, df])
        except Exception as ex:
            print(sha256, ex)
            pass
    if file_name is not None:
        merged_df.to_csv(file_name)
    return merged_df
