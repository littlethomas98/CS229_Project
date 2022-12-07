"""
Author : Baris Baran Gundogdu
"""

# importing panda library
import pandas as pd
import glob
import os.path
import sys
YEAR_DICT = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "June", "07": "July"
                                       ,"08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}
# COL_NAMES = ["column1", "column2", "column3", "column4", "column5", "column6", "column7", "column8",
#              "column9", "column10", "column11", "column12", "column13", "column14", "column15"]
COL_NAMES = [f"Column{i}" for i in range(1, 16)]
def main():
    for file in os.listdir(os.curdir):
        if file.endswith(".txt"):
            with open(file) as f:
                input_file = file
                parts = input_file.split('_')
                year = parts[6].strip()
                month = parts[7].split('.')[0]
                output_file = year + YEAR_DICT[month] + ".csv"
                # read the given file into a data frame
                dataframe = pd.DataFrame([line.strip().split() for line in f.readlines()])

                # store the data frame into csv file.
                dataframe.to_csv(output_file, index=False, header = COL_NAMES)


if __name__ == '__main__':
    main()