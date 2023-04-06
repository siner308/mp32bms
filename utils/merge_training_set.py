import pandas


def merge():
    training_set = pandas.read_csv('../training_set_five_seconds.csv')

    name = training_set['name']

    new_df = pandas.DataFrame()

    # each name
    for name in name.unique():
        # loc by name
        name_df = training_set.loc[training_set['name'] == name]

        # merge 5 rows to 1 row by concat columns
        # increase column names by 800 ex) output_columns_1 -> output_columns_801
        # increase onset names by 100 ex) input_onset_1 -> input_onset_101
        merged_row = pandas.DataFrame()

        for i, row in name_df.iterrows():
            # columns that starts with 'input_onset_' or 'output_columns_'
            onsets = row.loc[row.index.str.startswith('input_onset_')]
            columns = row.loc[row.index.str.startswith('output_columns_')]

            onset_offset = (i % 5) * 100
            column_offset = (i % 5) * 800

            merged_row['input_level'] = row['input_level']
            merged_row['name'] = row['name']

            # iter series columns
            for column in onsets.index:
                # increase onset names by 100 ex) input_onset_1 -> input_onset_101
                merged_row[column.replace(column.split('_')[2], str(int(column.split('_')[2]) + onset_offset))] = row[
                    column]

            for column in columns.index:
                # increase column names by 800 ex) output_columns_1 -> output_columns_801
                merged_row[column.replace(column.split('_')[2], str(int(column.split('_')[2]) + column_offset))] = row[
                    column]

            if (i + 1) % 5 == 0:
                new_df = new_df.append(merged_row)
                merged_row = pandas.DataFrame()

        # fill empty values and append if merged_row is not appended
        if not merged_row.empty:
            # fill input onset to 499
            for i in range(500):
                if 'input_onset_' + str(i) not in merged_row.columns:
                    merged_row['input_onset_' + str(i)] = 0

            # fill output columns to 3999
            for i in range(4000):
                if 'output_columns_' + str(i) not in merged_row.columns:
                    merged_row['output_columns_' + str(i)] = 0

            new_df = new_df.append(merged_row)

        print(name + ' is merged')

    # save merged df
    new_df.to_csv('./merged_training_set_five_seconds.csv', index=False)


if __name__ == "__main__":
    merge()
