#TODO: DELETE DONT USE IT IN OUR CODE   
class remove_elements(object):
    def __init__(self, cfg_data):
        self.input_list = cfg_data['input_list']
        self.number_elements = cfg_data['number_elements']

        self.input_list = self.input_list.sort_values(by=['user_id', 'timestamp'], ascending=[True, True])
   
    def remove_first_n_elements(self):
        remove_elements = self.input_list.groupby('user_id').apply(lambda x: x.iloc[self.number_elements:])
        return remove_elements
    
    def remove_last_n_elements(self):
        remove_elements = self.input_list.groupby('user_id').apply(lambda x: x.drop(x.tail(self.number_elements+1).index[:-1]))
        return remove_elements
    
    def remove_middle_n_elements(self):
        remove_elements = self.input_list.groupby('user_id').apply(lambda x: x.iloc[:self.number_elements].append(x.iloc[-self.number_elements:])) #non l'ho fatto così ma come sotto specificando l'indice da cui partire per rimuovere
        return remove_elements
    

'''
def remove_n_rows(df):
  if len(df) > index_to_remove + n:
    head = df.iloc[:index_to_remove]
    middle = df.iloc[index_to_remove:index_to_remove + n]
    tail = df.iloc[index_to_remove + n:]

    return pd.concat([head, tail])
  else:
    return df


# Apply the function to each group and concatenate the results
df_removed_middle = df.groupby('user_id:token').apply(remove_n_rows).reset_index(drop=True)

'''