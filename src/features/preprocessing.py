import pandas as pd
import numpy as np
import os
def give_me_downsample_data(at_sampling_freq: str = None,
                            df: pd.DataFrame = None,
                            save_file: bool = False,
                            parent_save_path: str = None) -> pd.DataFrame:
    if at_sampling_freq == 'weekly':
        print("Preprocessing on Weekly Data.......")
        df['date'] = pd.to_datetime(df['date'], format='mixed')  # Corrected format
        df['weekly'] = df['date'].dt.strftime('%Y-%m-%W')  # Changed to %U for week number
        weekly_accidents = df.groupby('weekly').size().reset_index(name='weekly_accident')

        if save_file:
            weekly_saving_file_name = os.path.join(parent_save_path, 'weekly_data.csv')
            print(f'Saving Weekly data in Directory: {weekly_saving_file_name}')
            weekly_accidents.to_csv(weekly_saving_file_name, index=False)

        return weekly_accidents

    elif at_sampling_freq == 'monthly':
        print("Preprocessing on Monthly Data.......")
        df['date'] = pd.to_datetime(df['date'], format='mixed')  # Corrected format
        df['monthly'] = df['date'].dt.strftime('%Y-%m')
        monthly_accidents = df.groupby('monthly').size().reset_index(name='monthly_accident')

        if save_file:
            monthly_saving_file_name = os.path.join(parent_save_path, 'monthly_data.csv')
            print(f'Saving Monthly data in Directory: {monthly_saving_file_name}')
            monthly_accidents.to_csv(monthly_saving_file_name, index=False)

        return monthly_accidents

    elif at_sampling_freq == 'daily':
        print("Preprocessing on Daily Data.......")
        df['date'] = pd.to_datetime(df['date'], format='mixed')  # Corrected format
        df['daily'] = df['date'].dt.strftime('%Y-%m-%d')
        daily_accidents = df.groupby('daily').size().reset_index(name='daily_accident')

        if save_file:
            monthly_saving_file_name = os.path.join(parent_save_path, 'daily_data.csv')
            print(f'Saving Monthly data in Directory: {monthly_saving_file_name}')
            daily_accidents.to_csv(monthly_saving_file_name, index=False)
        return daily_accidents
    else:
        print("Invalid sampling frequency specified.")
        return None
