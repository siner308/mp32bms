import pandas


def split():
    """split timeseries features about 1 seconds to 0.5 seconds"""
    df = pandas.read_csv('../training_set.csv')
    rows_count = len(df)

    front_df = pandas.DataFrame()
    back_df = pandas.DataFrame()

    for column in df.columns:
        # front df is having input_0 to input_49, not over input_50
        # front df is having output_0 to output_399
        if column.startswith('input_onset_'):
            if int(column.split('_')[2]) >= 50:
                # minus 50 from column name
                column = column.replace(column.split('_')[2], str(int(column.split('_')[2]) - 50))
                back_df[column] = df[column]
            else:
                front_df[column] = df[column]
        elif column.startswith('output_columns_'):
            if int(column.split('_')[2]) >= 400:
                # minus 400 from column name
                column = column.replace(column.split('_')[2], str(int(column.split('_')[2]) - 400))
                back_df[column] = df[column]
            else:
                front_df[column] = df[column]
        else:
            front_df[column] = df[column]
            back_df[column] = df[column]

    # save front df
    front_df.to_csv('./front_training_set.csv', index=False)

    # save back df
    back_df.to_csv('./back_training_set.csv', index=False)


if __name__ == "__main__":
    split()
