# Importing packages
import csv
import re
import pandas as pd
from itertools import combinations
 
# Function to check for INT Datatype
def is_integer(attr):
    try:
        int(attr)
        return True
    except ValueError:
        return False 

# Function to check for VARCHAR Datatype
def is_alphanumeric(attr):
    return bool(re.match("^[.a-zA-Z0-9]*$", attr))

# Function to check for Date Datatype
def is_date(attr):
    try:
        pd.to_datetime(attr)
        return True 
    except ValueError:
        return False 

# Function to check for Email Datatype
def is_email(attr):
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$'
    return re.match(email_pattern, attr)   # returns True if the attribute is an email address (@mst.edu/@mst.com/@gmail.com) else False

# Function to check the datatypes of each attribute
def check_datatypes(csv_filePath):
    import csv
    with open(f'{csv_filePath}', mode ='r', encoding='ISO-8859-1')as file:
        csvFile = csv.reader(file)
        header = next(csvFile)
        row1 = next(csvFile)
        data_types = {} # Dictionary type variable to store the datatype of each column
        for i,j in zip(header, row1):
            if(is_integer(j)):
                data_types[i] = "INT"
            elif(is_alphanumeric(j)):
                data_types[i] = "VARCHAR(100)"
            elif(is_date(j)):
                data_types[i] = "DATE"
            elif(is_email(j)):
                data_types[i] = "VARCHAR(50)"
    return data_types

# Function to sequentially check each normal form and identify the highest normal form of given table
def check_normal_form(csv_filePath, FD, Key, MVD):
    if not check_1NF(csv_filePath):
        return "Not in any Normal Form"

    if not check_2NF(FD, Key):
        return "In 1NF"
    
    if not check_3NF(FD, Key):
        return "In 2NF"
    
    if not check_BCNF(FD, Key):
        return "In 3NF"
    
    if not check_4NF(FD, Key, MVD):
        return "In BCNF"
    
    if not check_5NF(FD, Key, MVD):
        return "In 4NF"
    
    return "In 5NF"

# Function to identify if the given table is in 1NF
def check_1NF(csv_filePath):
    # Checks if the table is in 1NF by verifying that all cell values are atomic (i.e., no multi-valued attributes)
    # Open the CSV file in read mode
    with open(csv_filePath, mode='r', encoding='ISO-8859-1') as file:
        csv_reader = csv.reader(file)
        
        # Iterate over each row and cell
        for row in csv_reader:
            if not is_row_atomic(row):
                return False  # table have non-atomic values
    return True # table doesn't have non-atomic values

def is_row_atomic(row):

    for cell in row:
        # Strip whitespace to ensure no issues with spaces around commas
        cell = cell.strip()
        
        # Check if a comma exists within the cell
        if ',' in cell:
            return False
    return True 

# Function to identify if the given table is in 2NF
def check_2NF(FD, candidate_keys):
    for dependency in FD:
        determinant, dependent = dependency.split("->")
        
        # Check for partial dependency on each candidate key
        for key in candidate_keys:
            # If the determinant is a proper subset of a candidate key
            if set(determinant).issubset(set(key)) and set(determinant) != set(key):
                return False  # Partial dependency exists
    return True  # No partial dependency exists


def is_superkey(determinant, candidate_keys):
    # Check if the determinant is a superkey by comparing with candidate keys
    return any(set(determinant) == set(key) or set(determinant).issuperset(set(key)) for key in candidate_keys)

def check_3NF(FD, candidate_keys):
    # Create a set of all attributes that are part of any candidate key (prime attributes)
    prime_attributes = set(attr for key in candidate_keys for attr in key)

    # Traverse through each functional dependency
    for dependency in FD:
        # Splitting each functional dependency into lhs (determinant) and rhs (dependent)
        lhs, rhs = dependency.split('->')
        FD_l = [attr.strip() for attr in lhs.strip().split(',')]
        FD_r = [attr.strip() for attr in rhs.strip().split(',')]

        # Check if LHS is a superkey or RHS only contains prime attributes
        if not is_superkey(FD_l, candidate_keys):
            # If there are any non-prime attributes in the RHS, it's a violation of 3NF
            if any(attr not in prime_attributes for attr in FD_r):
                return False  # 3NF violation found

    return True  # No 3NF violations found


def check_BCNF(FD, candidate_keys):
    for dependency in FD:
        lhs, rhs = dependency.split("->")
        lhs = [attr.strip() for attr in lhs.strip().split(',')]

        # Check if the LHS of the functional dependency is a superkey
        if not is_superkey(lhs, candidate_keys):
            return False  # Found a BCNF violation

    return True  # No BCNF violations found

# Function to identify if the given table is in 4NF
def check_4NF(FD, Key, MVD):
    FDs = []
    MVDs = []
    # iterating through each functional dependency
    for fd in FD:
        lhs,rhs = fd.split("->")
        l = lhs.strip().split(',')
        k = []
        for i in l:
            k.append(i.strip())
        r = rhs.strip().split(',')
        for i in r:
            k.append(i.strip())
        # All the elements of a functional dependency are appended to FDs list
        FDs.append(sorted(k))
    mvd_dic = {}
    # Iterating though each Multi-Valued Dependency
    for mvd in MVD:
        lhs,rhs = mvd.split("->>")
        l = lhs.strip().split(',')
        kl = []
        for i in l:
            kl.append(i.strip())
        r = rhs.strip().split(',')
        kr = []
        for i in r:
            kr.append(i.strip())
        if(mvd_dic.get(kl[0])==None):
            mvd_dic[kl[0]] = []
            mvd_dic[kl[0]].extend(kl + kr)
        else:
            mvd_dic[kl[0]].extend(kr)
    for table, attributes in mvd_dic.items():
        # if A->> B and A->> C then [A,B,C] is added to MVDs list
        MVDs.append(sorted(list(set(attributes))))
    # Iterating though each Multi-Valued Dependency in MVDs
    for i in MVDs:
        # Iterating though each Functional Dependency in FDs
        for j in FDs:
            # If MVD is in FD then that means the given MVD is invalid as there is a relation between B and C
            # Else if it is not a subset return False
            if(not set(i).issubset(set(j))):
                return False # Multi-valued dependency exists
    return True # No multi-valued dependency exists

# Function to identify if the given table is in 5NF
def check_5NF(FD, Key, MVD):

    def is_superkey(key, fds):
    # Check if the given key is a superkey for the set of FDs
        for fd in fds:
            if set(fd[0].split(', ')).issubset(key):
                if not set(fd[1].split(', ')).issubset(key):
                    return False
        return True

    # Check for join dependencies using natural join.
    for mv in MVD:
        left, right = mv.split(' ->> ')
        left_set = set(left.split(', '))
        right_set = set(right.split(', '))
        # Check if both sides of the MVD are superkeys or candidate keys.
        if not (is_superkey(left_set, Key) or left_set in Key):
            return False
        if not (is_superkey(right_set, Key) or right_set in Key):
            return False
        # Check if the natural join of the MVD can be expressed using FDs.
        join_attrs = left_set & right_set  # Attributes common to both sides
        for i in range(1, len(join_attrs) + 1):
            # Check natural join for subsets of the common attributes.
            for subset in combinations(join_attrs, i):
                subset_set = set(subset)
                if not (subset_set in Key or is_superkey(subset_set, Key)):
                    return False

    return True

def calculate_total_combinations(lengths):
    
    total = 1
    for length in lengths:
        total *= length
    return total

def generate_combinations(num_elements, total_combinations):
    
    combinations = []
    for i in range(total_combinations):
        new_row = []
        for j in range(len(num_elements)):
            element_index = i % num_elements[j]
            new_row.append(num_elements[j][element_index])
            i //= num_elements[j]
        combinations.append(new_row)
    return combinations

import pandas as pd

def convert_to_1NF(csv_filePath):  
    # Read the CSV file into a DataFrame
    df = pd.read_excel(csv_filePath)
    
    # Initialize a DataFrame to store the normalized result
    normalized_df = df.copy()

    # Identify columns that violate 1NF and normalize them
    for column in df.columns:
        # Check for 1NF violation: if any value is comma-separated
        if df[column].apply(lambda x: isinstance(x, str) and ',' in x).any():
            # Split values by ',' and expand rows
            df_expanded = df[column].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
            df_expanded = df_expanded.str.strip()  # Remove extra spaces

            # Drop the original violating column from the normalized DataFrame
            normalized_df = normalized_df.drop(column, axis=1)

            # Join the expanded column back to the main DataFrame with other columns repeated
            normalized_df = normalized_df.join(df_expanded.rename(column))

    # Reset index to ensure it aligns with a properly normalized structure
    normalized_df = normalized_df.reset_index(drop=True)

    return normalized_df


# Function to convert the given table to 2NF
def is_proper_subset(subset, superset):
    """Check if subset is a proper subset of superset."""
    return set(subset).issubset(set(superset)) and set(subset) != set(superset)

def convert_to_2NF(FD, candidate_keys):
    tables = {}
    main_table_attributes = set()
    
    # Traverse each functional dependency
    for fd in FD:
        # Split FD into LHS (determinant) and RHS (dependent)
        lhs, rhs = fd.split('->')
        lhs = [attr.strip() for attr in lhs.split(',')]
        rhs = [attr.strip() for attr in rhs.split(',')]
        
        # Determine if LHS is a proper subset of any candidate key
        partial_dependency = False
        for key in candidate_keys:
            if is_proper_subset(lhs, key):
                partial_dependency = True
                break
        
        if partial_dependency:
            # Create a new table for the partial dependency
            table_name = "_".join(lhs)
            if table_name not in tables:
                tables[table_name] = set(lhs + rhs)
            else:
                tables[table_name].update(rhs)
        else:
            # Add attributes to main table
            main_table_attributes.update(lhs + rhs)
    
    # Ensure candidate keys are in the main table
    main_table_name = "Main_Table"
    tables[main_table_name] = main_table_attributes

    # Convert each set of attributes in tables to lists to maintain consistency
    tables = {table: list(attributes) for table, attributes in tables.items()}

    return tables


def is_superkey(determinant, candidate_keys):
    # Check if the determinant is a superkey
    return any(set(determinant) == set(key) or set(determinant).issuperset(set(key)) for key in candidate_keys)

def convert_to_3NF(FD, candidate_keys, tables):
    new_tables = {}

    for fd in FD:
        # Split the left-hand side (LHS) and right-hand side (RHS) of the FD
        lhs, rhs = fd.split('->')
        lhs = [attr.strip() for attr in lhs.split(',')]
        rhs = [attr.strip() for attr in rhs.split(',')]

        # Check for transitive dependency: if LHS is not a superkey but RHS is non-prime
        if not is_superkey(lhs, candidate_keys):
            # Create a new table for the transitive dependency
            table_name = "_".join(lhs)
            if table_name not in new_tables:
                new_tables[table_name] = set(lhs + rhs)
            else:
                new_tables[table_name].update(rhs)
        else:
            # Add the FD to an existing or new main table if LHS is a superkey
            main_table_name = "Main_Table"
            if main_table_name not in new_tables:
                new_tables[main_table_name] = set(lhs + rhs)
            else:
                new_tables[main_table_name].update(rhs)

    # Remove duplicate attributes in each table
    for table, attributes in new_tables.items():
        new_tables[table] = list(attributes)

    return new_tables

# Function to convert the given table to BCNF
def convert_to_BCNF(FD, candidate_keys, tables):
    new_tables = {}

    # Traverse through each functional dependency
    for fd in FD:
        lhs, rhs = fd.split('->')
        lhs_fd = [attr.strip() for attr in lhs.split(',')]
        rhs_fd = [attr.strip() for attr in rhs.split(',')]

        # If LHS is not a superkey, we need to decompose
        if not is_superkey(lhs_fd, candidate_keys):
            # Create a new table for this dependency
            new_table_name = "_".join(lhs_fd)  # Use a meaningful name based on LHS attributes
            new_tables[new_table_name] = lhs_fd + rhs_fd
            
            # Ensure the candidate table retains the key and the LHS
            if "Candidate" not in new_tables:
                new_tables["Candidate"] = list(candidate_keys[0])  # Assuming there's at least one candidate key
            new_tables["Candidate"].extend(lhs_fd)

        # If LHS is a superkey, add the attributes to the candidate table
        else:
            if "Candidate" not in new_tables:
                new_tables["Candidate"] = list(candidate_keys[0])  # Assuming there's at least one candidate key
            new_tables["Candidate"].extend(rhs_fd)  # Include RHS to the candidate table

    # Remove duplicate attributes in each table
    for table, attributes in new_tables.items():
        new_tables[table] = list(set(attributes))

    return new_tables
# Function to convert the given table to 4NF
def convert_to_4NF(FD, Key, MVD, tables):
    # highest integer value in keys in given tables dictionary
    # to get the next integer value for a table name in new tables
    highest_int = 0
    for key in tables.keys():
        if(isinstance(key, int)):
            if(key>highest_int):
                highest_int = key
    new_tables = {}
    mvds = {}
    #traversing through each MVD
    for mvd in MVD:
        # splitting each mvd of MVD to lhs and rhs
        lhs,rhs = mvd.split('->>')
        l = lhs.strip().split(',')
        lhs = []
        for i in l:
            lhs.append(i.strip())
        r = rhs.strip().split(',')
        rhs = []
        for i in r:
            rhs.append(i.strip())
        # Appending all MVDs to mvds dictionary
        if(lhs[0] in mvds.keys()):
            mvds[lhs[0]].extend(rhs)
        else:
            mvds[lhs[0]] = []
            mvds[lhs[0]].extend(lhs+rhs)
    # traversing through each value of MVD dictionary
    for items in mvds.values():
        flag = 0
        # traversing through each relation of tables dictionary
        for attr in tables.values():
            # if the mvd exists in attr that is X->A and X->B if there is a relation (X,A,B) then the MVD is invalid
            if(set(items).issubset(set(attr))):
                flag = 1
        
    # Merging both original and new dictionary
    merged_dict = {**tables, **new_tables}    

    # travering through each table-attribute pair, to get only distinct attribute pairs in the table
    for table, attributes in merged_dict.items():
        c = set()
        for i in attributes:
            c.add(i)
        merged_dict[table] = list(c)
        
    return merged_dict


def detect_join_dependencies(FD, Key, MVD):
    join_dependencies = []
    
    # Iterate over FD to identify join dependencies based on MVDs
    for fd in FD:
        lhs, rhs = fd.split("->")
        lhs_attributes = lhs.strip().split(",")
        rhs_attributes = rhs.strip().split(",")
        
        # A join dependency is valid if the LHS and RHS have multiple attributes and are not trivial dependencies
        if len(lhs_attributes) > 1 or len(rhs_attributes) > 1:
            join_dependencies.append(fd.strip())
            
    return join_dependencies

# Function to check if tables are in 5NF
# def check_5NF(FD, Key, MVD):
#     join_dependencies = detect_join_dependencies(FD, Key, MVD)
#     return len(join_dependencies) > 0

# Function to convert the given tables to 5NF
def convert_to_5NF(FD):
    # Detect join dependencies
    join_dependencies = detect_join_dependencies(FD)
    
    new_tables = {}
    table_counter = 1
    df = pd.read_csv(csv_filePath)
    
    # Decompose based on detected join dependencies
    for jd in join_dependencies:
        lhs, rhs = jd.split("->")
        lhs_attributes = [attr.strip() for attr in lhs.split(",")]
        rhs_attributes = [attr.strip() for attr in rhs.split(",")]
        
        # Create new table with lhs and rhs attributes
        table_name = f"DecomposedTable_{table_counter}"
        selected_columns = lhs_attributes + rhs_attributes
        new_tables[table_name] = df[selected_columns].drop_duplicates().reset_index(drop=True)
        
        # Remove RHS columns from the main table to avoid redundancy
        df = df.drop(columns=rhs_attributes)
        
        table_counter += 1
    
    # Add the remaining (reduced) original table if any columns are left
    if not df.empty:
        new_tables["RemainingTable"] = df.drop_duplicates().reset_index(drop=True)

    return new_tables


# Function to generate SQL queries
def format_sql_query(query):
    """
    Format a SQL query to make it more readable with proper indentation and line breaks.
    
    Args:
        query (str): The SQL query to format
        
    Returns:
        str: The formatted SQL query
    """
    # Remove extra spaces
    query = ' '.join(query.split())
    
    # Add newline after CREATE TABLE
    query = query.replace('CREATE TABLE', 'CREATE TABLE\n  ')
    
    # Add newline and indent after opening parenthesis
    query = query.replace(' (', ' (\n    ')
    
    # Add newlines and indents for column definitions
    parts = query.split(',')
    formatted_parts = []
    
    for i, part in enumerate(parts):
        part = part.strip()
        
        # Handle PRIMARY KEY constraint
        if 'PRIMARY KEY' in part and '(' in part:
            pk_part = part.split('PRIMARY KEY')
            if len(pk_part) > 1:
                formatted_parts.append(pk_part[0].strip())
                formatted_parts.append('    PRIMARY KEY' + pk_part[1].strip())
            else:
                formatted_parts.append(part)
        
        # Handle FOREIGN KEY constraints
        elif 'FOREIGN KEY' in part:
            formatted_parts.append('    ' + part)
            
        # Normal column definitions
        else:
            # Don't add indent for the first part (it's already indented)
            if i == 0:
                formatted_parts.append(part)
            else:
                formatted_parts.append('    ' + part)
    
    # Join parts back together
    query = ',\n'.join(formatted_parts)
    
    # Add proper closing
    query = query.replace(');', '\n);')
    
    return query

def format_sql_statements(sql_statements):
    """
    Format a list of SQL statements with proper spacing between statements.
    
    Args:
        sql_statements (list): List of SQL statements to format
        
    Returns:
        str: Formatted SQL statements as a single string
    """
    formatted_statements = []
    
    for stmt in sql_statements:
        formatted_stmt = format_sql_query(stmt)
        formatted_statements.append(formatted_stmt)
    
    # Join statements with two newlines between them
    return '\n\n'.join(formatted_statements)

# Modify the generate_sql_queries function to use the formatter
def generate_sql_queries(FD, Key, tables, data_types):
    sql_statements = []
    fd_lhs = []
    fd_rhs = []
    lhs_fd = []
    
    for fd in FD:
        l,r = fd.split("->", 1)
        l1= l.strip().split(",")
        lhs_fd.extend(l1)
    
    for fd in FD:
        lhs, rhs = fd.split('->')
        x = lhs.strip().split(',')
        for i in x:
            if(i not in fd_lhs):
                fd_lhs.append(i)
        y = rhs.strip().split(',')
        for i in y:
            if(i not in fd_rhs):
                fd_rhs.append(y)

    for table_name, columns in tables.items():
        foreign_query = ""
        query = f'CREATE TABLE {table_name} ('
    
        count_of_keys = sum(1 for x in columns if x in fd_lhs)

        for i, attr in enumerate(columns):
            query += f'{attr} {data_types.get(attr, "VARCHAR(255)")}' 
            if attr not in lhs_fd:
                query += " NOT NULL"
            if count_of_keys == 1 and attr in fd_lhs:
                query += " PRIMARY KEY"
            elif attr == table_name:
                query += " PRIMARY KEY"
            elif attr in fd_lhs:
                foreign_query += f', FOREIGN KEY ({attr}) REFERENCES {attr}({attr})'
            if i != (len(columns) - 1):
                query += ", "
        
        xl = ""
        if table_name == "Candidate":
            xl += ', PRIMARY KEY ('
            for z in range(len(Key)):
                xl += f'{Key[z]}'
                if z < (len(Key) - 1):
                    xl += ","
            xl += ")"
        if xl:
            query += xl
        if foreign_query:
            query += foreign_query
        query += ");"
    
        sql_statements.append(query)
    
    # Format the SQL statements
    formatted_sql = format_sql_statements(sql_statements)
    
    # Print the formatted SQL
    print("\nGenerated SQL Statements:")
    print(formatted_sql)
    
    return sql_statements


# Input commands
#csv_filePath = input("Enter input filepath: ")
csv_filePath = "C:/Users/boddu/OneDrive/Desktop/dbproject/StudentCourses.csv"



# Taking input for functional dependencies
FD = []
print("Enter Functional Dependencies in the format: A->B; A,B->C; A->B,C")

# Take input as a single line and split by semicolons
fd_input = input()
FD = [fd.strip() for fd in fd_input.split(';') if fd.strip()]

print("Functional Dependencies:", FD)


# Taking input for multi-valued dependencies
MVD = []
print("Enter Multi Valued Dependencies in the format: A->>B; A->>C")

# Take input as a single line and split by semicolons
mvd_input = input()
MVD = [mvd.strip() for mvd in mvd_input.split(';') if mvd.strip()]

print("Multi Valued Dependencies:", MVD)

# Taking input for Primary Key
print("Enter Key:")
Key = input().split(",")

# Taking input if the user wants to Find the highest normal form of the input table? (1: Yes, 2: No):
input_normal_form = "The given table is "
print("Find the highest normal form of the input table? (1: Yes, 2: No):")
print("Enter 1 or 2")
x = int(input())
if(x==1):
    input_normal_form += check_normal_form(csv_filePath, FD, Key, MVD)

# Taking input from user to get the Choice of the highest normal form to reach (1: 1NF, 2: 2NF, 3: 3NF, B: BCNF, 4: 4NF, 5: 5NF):
print("Choice of the highest normal form to reach (1: 1NF, 2: 2NF, 3: 3NF, B: BCNF, 4: 4NF, 5: 5NF):")
print("Enter 1/2/3/B/4/5")
k = input()
user_choice = 0
# For further usage of the variable if the user inputs to convert the table to BCNF the input B is converted to "3.5"
if(k == '1'):
    user_choice = 1
if(k == '2'):
    user_choice = 2
if(k == '3'):
    user_choice = 3
if(k == 'B' or k == 'b'):
    user_choice = 3.5
if(k == '4'):
    user_choice = 4
if(k == '5'):
    user_choice = 5

result_1NF = []


if(user_choice >= 1):
    if(not check_1NF(csv_filePath)):
        result_1NF = convert_to_1NF(csv_filePath)
        print("Enter output csv file path to store the result after converting to 1NF")
        # Open the .csv file in write mode
        with open('converted1NF.csv', mode='w', newline='') as file:
            # Create a csv.writer object
            writer = csv.writer(file)
        # Write the data to the .csv file
            for row in result_1NF:
                writer.writerow(row)
res_tables = {}
# based on k the functions from convert_to_1NF to convert_to_(k)NF will be executed
if(user_choice >= 2):
    res_tables = convert_to_2NF(FD, Key)
if(user_choice >= 3):
    res_tables = convert_to_3NF(FD, Key, res_tables)
if(user_choice >= 3.5):
    res_tables = convert_to_BCNF(FD, Key, res_tables)
if(user_choice >= 4):
    res_tables = convert_to_4NF(FD, Key, MVD, res_tables)
if(user_choice == 5):
    res_tables = convert_to_5NF(FD)

# Storing the data types of each variable
data_types = check_datatypes(csv_filePath)

# Function call to generate SQL queries for the new decomposed relations
SQL_queries = generate_sql_queries(FD, Key, res_tables, data_types)
print(SQL_queries)

# Loading the results to output.txt file
with open('Output.txt', mode='w', newline='') as file:
    # Create a csv.writer object
    writer = csv.writer(file, delimiter=" ")

# If the input table is not in 1NF, write the converted 1NF table to the file
    if len(result_1NF) > 0:
        for row in result_1NF:
            writer.writerow(row)  # Writing each row in result_1NF as CSV
        writer.writerow([])  
    
    # Write the SQL queries to the files
    for query in SQL_queries:
        writer.writerow(query)  # Writing each query as a row in the output
    writer.writerow([])  # Write empty rows as a separator
    
    # Writing the highest normal form of the input table, if required
    if x == 1:
        writer.writerow([input_normal_form])  # Writing the input normal form