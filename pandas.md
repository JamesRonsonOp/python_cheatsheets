# Pandas CheatSheet

### Imports

`import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline`

* properties do not have () after whereas methods/functions have a () *


### Reading in Data
**Use these commands to import data from a variety of different sources and formats.**

|Code Example| Description |
|-----|-----|
|`pd.read_csv(filename)` | From a CSV file |
| `pd.read_csv(‘filename.csv’, usecols = [“specify col to import”))` | importing csv but specifying the columns to import | 
| `pd.read_table(filename)` | From a delimited text file (like TSV) |
|`pd.read_excel(filename)` | From an Excel file |
| `pd.read_sql(query, connection_object)` | Read from a SQL table/database |
| `pd.read_json(json_string)` | Read from a JSON formatted string, URL or file. |
|`pd.read_html(url)` | Parses an html URL, string or file and extracts tables to a list of dataframes |
| `pd.read_clipboard()` | Takes the contents of your clipboard and passes it to read_table() |
| `pd.DataFrame(dict)` | From a dict, keys for columns names, values for data as lists |

### Exporting Data
* Use these commands to export a DataFrame to CSV, .xlsx, SQL, or JSON. *

|Code Example| Description |
|-----|-----|
|`df.to_csv(filename)` | Write to a CSV file | 
|`df.to_excel(filename)` | Write to an Excel file |
|`df.to_sql(table_name, connection_object)` | Write to a SQL table |
|`df.to_json(filename)` | Write to a file in JSON format |

### Create Test Objects
*These commands can be useful for creating test segments.*
|Code Example| Description |
|-----|-----|
|`pd.DataFrame(np.random.rand(20,5))` | 5 columns and 20 rows of random floats|
| `pd.Series(my_list)` | Create a series from an iterable my_list |
| `df.index = pd.date_range('1900/1/30', periods=df.shape[0])` | Add a date index |

### Viewing/Inspecting Data
*Use these commands to take a look at specific sections of your pandas DataFrame or Series.*

|Code Example| Description |
|-----|-----|
|`df.head(n)` | First n rows of the DataFrame |
|`df.tail(n)` | Last n rows of the DataFrame |
|`df.shape` | Number of rows and columns |
|`df.info()` | Index, Datatype and Memory information |
|`df.describe()` | Summary statistics for numerical columns|
|`s.value_counts(dropna=False)` | View unique values and counts|
|`df.apply(pd.Series.value_counts)`| Unique values and counts for all columns|
|`df.index()` | Shows row index of dataframe |
|`s.unique()` | returns an array of unique objects from a series |
|`s.nunique()` | returns the number of unique objects |


### Selection/Indexing
* Use these commands to select a specific subset of your data. *

df[col] | Returns column with label col as Series
df[[col1, col2]] | Returns columns as a new DataFrame
s.iloc[0] | Selection by position
s.loc['index_one'] | Selection by index
df.iloc[0,:] | First row
df.iloc[0,0] | First element of first column
df.sort_index() | Returns a new Df sorted by label if inplace = False. Else updates orig. Df. 
df.swaplevel() | Swap levels i and j in a MultiIndex on a particular axis. 
df.reset_index() | Reset index of Df and use the default one instead. 
df.reset_index() *multi- index | Remove one or more levels of the multindex
df.unstack() | AKA pivot. Series with MultiIndex to produce DF. Level involved will get sorted.



Data Cleaning
Use these commands to perform a variety of data cleaning tasks.
df.columns = ['a','b','c'] | Rename columns
pd.isnull() | Checks for null Values, Returns Boolean Arrray
df.isna() | Checks for null Values, Returns Boolean Arrray
df.isnull() | returns true or false for NaN values. 
index.isnull() | returns true or false for NaN values
index.notna() | returns true or false for non-NaN values
index.dropna() |Omit entries with missing values. 
series.isnull() | Detect missing values in a Series object. 
pd.notnull() | Opposite of pd.isnull()
df.dropna() | Drop all rows that contain null values
df.dropna(axis=1) | Drop all columns that contain null values
df.dropna(axis=1,thresh=n) | Drop all rows have have less than n non null values
df.drop(columns = ['column1', 'column2'])|remove rows or columns
df.fillna(x) | Replace all null values with x
s.fillna(s.mean()) | Replace all null values with the mean (mean can be replaced with almost any function from the statistics module)
s.astype(float) | Convert the datatype of the series to float
s.replace(1,'one') | Replace all values equal to 1 with 'one'
s.replace([1,3],['one','three']) | Replace all 1 with 'one' and 3 with 'three'
df.rename(columns=lambda x: x + 1) | Mass renaming of columns
df.rename(columns={'old_name': 'new_ name'}) | Selective renaming
df.set_index('column_one') | Change the index
df.rename(index=lambda x: x + 1) | Mass renaming of index
df[[‘column1’, ‘column2’, ‘column3’]] | Reorder columns by heading. 



Filter, Sort, and Groupby
Use these commands to filter, sort, and group your data.
df[df[col] > 0.5] | Rows where the column col is greater than 0.5
df[(df[col] > 0.5) & (df[col] < 0.7)] | Rows where 0.7 > col > 0.5
df.sort_values(col1) | Sort values by col1 in ascending order
df.sort_values(col2,ascending=False) | Sort values by col2 in descending order
df.sort_values([col1,col2],ascending=[True,False]) | Sort values by col1 in ascending order then col2 in descending order
df.groupby(col) | Returns a groupby object for values from one column
df.groupby([col1,col2]) | Returns groupby object for values from multiple columns
df.groupby(col1)[col2] | Returns the mean of the values in col2, grouped by the values in col1 (mean can be replaced with almost any function from the statistics module)
df.pivot_table(index=col1,values=[col2,col3],aggfunc=mean) | Create a pivot table that groups by col1 and calculates the mean of col2 and col3
df.groupby(col1).agg(np.mean) | Find the average across all columns for every unique col1 group
df.apply(np.mean) | Apply the function np.mean() across each column
df.apply(np.max,axis=1) | Apply the function np.max() across each row
df.transform() | Create another Df with same index as passed Df but with transformed values.
df.apply(lambda x: x * a, axis= 1) | Apply the function will allow us to pass a function over an axis of our DataFrame. 

Manually Creating a Filter


Join/Combine/Insert
Use these commands to combine multiple dataframes into a single one.
df1.append(df2) | Add the rows in df1 to the end of df2 (columns should be identical)
pd.concat([df1, df2],axis=1) | Add the columns in df1 to the end of df2 (rows should be identical)
df1.join(df2,on=col1,how='inner') | SQL-style join the columns in df1 with the columns on df2 where the rows for col have identical values. 'how' can be one of 'left', 'right', 'outer', 'inner'. No need to specify columns.
df1.merge() | like join, merge does not function across rows. Additionally, you will note that without a common column to merge on, you must be explicit about telling pandas to merge on the fight and left indices. without a common column to merge on, you must be explicit about telling pandas to merge on the right and left indices. You can do this with left_index=True and right_index=True. Because we are not merging on a column, we do not pass on or how parameters.
Df[‘name_of_new_inserted_column’] = values | Inserts new column into df. The values can be many things. A pd.series, a list or array, and a single value for all rows. 

Statistics
Use these commands to perform various statistical tests. (These can all be applied to a series as well.)

df.describe() | Summary statistics for numerical columns
df.mean() | Returns the mean of all columns (average)
df.corr() | Returns the correlation between columns in a DataFrame
df.count() | Returns the number of non-null values in each DataFrame column
df.max() | Returns the highest value in each column
df.min() | Returns the lowest value in each column
df.median() | Returns the median of each column (value separating higher half and lower half)
df.std() | Returns the standard deviation of each column
df.nlargest() | Returns the first ‘n’ rows ordered by ‘columns’ in desc. Order. 
df.quantile() | The mean of the value of the data below the specified ‘q’ percentages in data
s.value_counts(normalize=True) | provides percentages of specific values occurring

Comparing/Matching/Extracting Data
Use these commands to perform comparisons. 

s.str.contains() | test if pattern is in a string of a series or index
s.str.startswith() | test if pattern is
s.str.endswith() | test if pattern is
s.str.contains() | test if pattern is
s.str.match() | test if pattern is
s.str.extract() | extract capture groups in the regex pat as DF columns 
s.str.extractall() | extract capture groups in the regex pat as DF columns
s.nunique() | return # of unique elements in the object. Excludes NA by default. 
df.nunique() | return series with number of distinct observations. Can ignore NaN values


Indexing
Use these commands to index
df.sort_index() | Returns a new Df sorted by label if inplace = False. Else updates orig. Df. 
df.swaplevel() | Swap levels i and j in a MultiIndex on a particular axis. 
df.reset_index() | Reset index of Df and use the default one instead. 
df.reset_index() *multi- index | Remove one or more levels of the multindex.
 
Creating a Dataframe
Method for Creating a Pandas Dataframe
Method 1 | Follow this generic template.

  

Pandas.map()

Used to map values from two series having one column same. For mapping two series, the last column of the first series should be same as index column of the second series, also the values should be unique. 

Series.map(arg, na_action = none)

Parameters: 
Arg: function, dict or series
Na_action: {none, ‘ignore’} if ‘ignore’, propagate NA values, without passing them to the mapping corre
![image](https://user-images.githubusercontent.com/57559181/133021843-9a871a50-f965-4ea3-b6e5-b2a09ed27c6a.png)

