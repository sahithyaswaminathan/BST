#!/usr/bin/env python
# coding: utf-8

# In[1]:

from functools import wraps
import os

# importing libraries
import pandas as pd
import numpy as np
import pytz
import time
import boto3

from utils import parallel_feature_calculation


BUCKET = 'pp-machine-learning-project'

DATA_DIR = 'Pending-Parcel-Rescrape-Project/data/'

# creating a dict for timezones and countries in our database
COUNTRY_TIMEZONE_DICT = {'0': 'Europe/London', 'Australia': 'Australia/Melbourne', 'United Kingdom': 'Europe/London', 'Germany': 'Europe/Berlin', 'Singapore': 'Asia/Singapore', 'Norway': 'Europe/Oslo', 'Israel': 'Africa/Blantyre', 'Philippines': 'Asia/Manila', 'Hong Kong': 'Asia/Hong_Kong', 'Canada': 'America/Toronto', 'Malaysia': 'Asia/Kuala_Lumpur', 'Netherlands': 'Europe/Amsterdam', 'Poland': 'Europe/Warsaw', 'China': 'Asia/Shanghai', 'Hungary': 'Europe/Budapest', 'Belgium': 'Europe/Brussels', 'New Zealand': 'Pacific/Auckland', 'India': 'Asia/Kolkata', 'Sweden':'Europe/Stockholm', 'Lithuania':'Europe/Vilnius', 'Austria':'Europe/Vienna', 'Italy':'Europe/Rome', 'Mexico':'America/Mexico_City', 'Switzerland':'Europe/Zurich', 'United States':'America/New_York', 'Spain':'Europe/Madrid', 'Slovenia':'Europe/Ljubljana', 'United Arab Emirates':'Asia/Dubai', 'Colombia':'America/Bogota', 'Vietnam':'Asia/Ho_Chi_Minh', 'Ukraine':'Europe/Kiev', 'Denmark':'Europe/Copenhagen', 'South Africa':'Africa/Johannesburg', 'Bulgaria':'Europe/Sofia', 'Portugal':'Europe/Lisbon', 'France':'Europe/Paris', 'Finland':'Europe/Helsinki', 'Cyprus':'Asia/Nicosia', 'Lebanon':'Asia/Beirut', 'Turkey':'Europe/Istanbul', 'Japan':'Asia/Tokyo', 'Estonia':'Europe/Tallinn', 'Luxembourg':'Europe/Luxembourg', 'Kazakhstan':'Asia/Almaty', 'South Korea':'Asia/Seoul', 'Belarus':'Europe/Minsk', 'Kuwait':'Asia/Kuwait', 'Serbia':'Europe/Belgrade', 'Argentina':'America/Argentina/Buenos_Aires', 'Thailand':'Asia/Bangkok', 'Sri Lanka':'Asia/Colombo', 'French Polynesia':'Pacific/Apia', 'Peru':'America/Lima', 'Indonesia':'Asia/Jakarta', 'Oman':'Asia/Muscat', 'Brazil':'America/Sao_Paulo', 'Jordan':'Asia/Amman', 'Czech Republic':'Europe/Prague', 'Greece':'Europe/Athens', 'Bahrain':'Asia/Bahrain', 'Saudi Arabia':'Asia/Riyadh', 'Romania':'Europe/Bucharest', 'Czechia':'Europe/Prague', 'Ireland':'Europe/Dublin', 'Latvia':'Europe/Riga', 'Russia':'Europe/Moscow', 'Costa Rica':'America/Costa_Rica', 'Papua New Guinea':'Pacific/Port_Moresby', 'Mauritius':'Indian/Mauritius', 'Gabon':'Africa/Libreville', 'Chile':'America/Santiago', 'Slovakia':'Europe/Bratislava', 'Croatia':'Europe/Zagreb', 'Kenya':'Africa/Nairobi', 'Nigeria':'Africa/Lagos', 'Maldives':'Indian/Maldives', 'Afghanistan':'Asia/Kabul', 'Swaziland':'Africa/Mbabane', 'Libya':'Africa/Tripoli', 'Uruguay':'America/Montevideo', 'Cambodia':'Asia/Phnom_Penh', 'Azerbaijan':'Asia/Baku', 'Zimbabwe':'Africa/Harare', 'Taiwan':'Asia/Taipei', 'Malta':'Europe/Malta', 'Jersey':'Europe/Jersey', 'Tonga':'Pacific/Tongatapu', 'Madagascar':'Indian/Antananarivo', 'New Caledonia':'Pacific/Noumea', 'Georgia':'Asia/Tbilisi', 'Tanzania':'Africa/Dar_es_Salaam', 'Egypt': 'Africa/Cairo', 'Qatar': 'Asia/Qatar', 'Iceland': 'Atlantic/Reykjavik', 'Ghana': 'Africa/Accra', 'Jamaica': 'America/Jamaica', 'Macau': 'Asia/Macau', 'Fiji': 'Pacific/Fiji', 'Iran': 'Asia/Tehran', 'Mozambique': 'Africa/Maputo', 'U.S. Virgin Islands': 'America/Antigua', 'Barbados': 'America/Blanc-Sablon', 'Reunion': 'America/Blanc-Sablon', 'Cameroon': 'Africa/Algiers', 'Togo': 'Africa/Lome', 'Puerto Rico': 'America/Puerto_Rico', 'Zambia': 'Africa/Lusaka'}

# dict containing country: timezone obj
TZ_TABLE = {country: pytz.timezone(tz_str) for country, tz_str in COUNTRY_TIMEZONE_DICT.items()}


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f'{func.__name__} called')
        resp = func(*args, **kwargs)
        # if str(os.environ.get('DEBUG')).lower() == 'true':
        print(f'{func.__name__} finished in {time.time() - start} seconds')
        return resp
    return wrapper


# In[59]:

def upload_data_to_s3(data, input_file_name, s3_dir=DATA_DIR, bucket=BUCKET):
    # saving data to csv
    pd.DataFrame(data).to_csv(input_file_name, index=False)

    s3_key = s3_dir + input_file_name

    # saving to S3
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(input_file_name, bucket, s3_key)


def import_data(path=''):
    # ### Importing data

    # In[59]:


    start = time.time()
    if os.path.isfile(path):
        input_files = [path]
    elif os.path.isdir(path):
        input_files = [os.path.join(path, f) for f in os.listdir(path)]
    else:
        raise Exception("Path specified is not a file nor a directory: {}".format(path))

    data = pd.concat((pd.read_csv(f, engine='c') for f in input_files), axis=0)
    end = time.time()

    print("Time taken to read and concatenate input files was {} seconds".format(end - start))

    # dropping row with headers repeated & fixing index
    start = time.time()
    data.drop(data[data.pk == 'pk'].index[0], inplace=True)

    data = data.reindex()

    data_og = data.copy()
    print("Time taken to 'dropping row with headers repeated & fixing index' was {} seconds".format(time.time() - start))
    return data, data_og


def add_utc_time(row):
    """
    Convert `first_event_time` to UTC (from local timezone)
    :param row: a row in a dataframe
    :return: time object with UTC tz
    """
    # updating time format to be UTC for first_event_time + timezone_awareness = location_local_time
    try:
        localtimezone = TZ_TABLE.get(row.origin_country)
        local_time = localtimezone.localize(row.first_event_time)
        #local_time = row.first_event_time.replace(tzinfo=localtimezone)
        utc_time = local_time.astimezone(pytz.utc)
    except Exception as err:
        print(err)
        utc_time = 'NA'
    return utc_time


def apply_first_event_time_utc(df):
    """Add new column `first_event_time_utc` to dataframes, with values derived from `add_utc_time()`"""
    df['first_event_time_utc'] = df.apply(add_utc_time, axis=1)
    return df


def preprocess_data(data, should_upload_to_s3=False, file_name=None,
                    num_processes=None, num_partitions=None):
    # ### Cleaning Data

    # In[60]:


    # removing all null first_event_time values
    data = data[data.first_event_time.notnull()]


    # In[61]:


    # adjusting fields within data
    # filling blank values in origin country with 0
    data.origin_country = data.origin_country.replace(to_replace=(np.NaN, "", " "), value='0')

    # removing spaces before or after Name
    data.origin_country = data.origin_country.str.strip()

    # Making US consistent
    data.loc[(data.origin_country == 'United States Of America'), 'origin_country'] = 'United States'


    # In[62]:


    ## Use for measuring impact of assumptions
    total_null_origins = data[data.origin_country == '0'].shape[0]


    # ### Updating null origin_county values based on assumptions

    # In[63]:


    ### OLD ASSUMPTIONS
    # updating missing country origin names in data
    data.loc[(data.origin_country == '0') & (data.org_name == 'Wayfair') & (data.carrier_name == 'UK Mail'), 'origin_country'] = 'United Kingdom'

    data.loc[(data.origin_country == '0') & (data.org_name == 'LOVEBONITO SINGAPORE PTE. LTD.'), 'origin_country'] = 'Singapore'

    data.loc[(data.origin_country == '0') & (data.org_name == 'Catch of the Day'), 'origin_country'] = 'Australia'

    data.loc[(data.origin_country == '0') & (data.org_name == 'Reliant / Entrego'), 'origin_country'] = 'Philippines'


    # In[64]:


    ### NEW ASSUMPTIONS
    data.loc[(data.origin_country == '0') & (data.org_name == 'Catch.com.au'), 'origin_country'] = 'Australia'

    data.loc[(data.origin_country == '0') & (data.org_name == 'Nespresso Netherlands'), 'origin_country'] = 'Netherlands'

    data.loc[(data.origin_country == '0') & (data.carrier_name == 'Swiss Post'), 'origin_country'] = 'Switzerland'

    data.loc[(data.origin_country == '0') & (data.carrier_name == 'DHL Paket'), 'origin_country'] = 'Germany'

    data.loc[(data.origin_country == '0') & (data.carrier_name == 'Colissimo'), 'origin_country'] = 'France'

    data.loc[(data.origin_country == '0') & (data.carrier_name == 'Singapore Post / Speedpost'), 'origin_country'] = 'Singapore'

    data.loc[(data.origin_country == '0') & (data.carrier_name == 'Ninja Van Indonesia'), 'origin_country'] = 'Indonesia'

    data.loc[(data.origin_country == '0') & (data.carrier_name == 'GLS Italy'), 'origin_country'] = 'Italy'


    # In[65]:


    # converting origin_country to string
    data.origin_country = data.origin_country.astype('str')

    # removing spaces before or after Name again incase it's caused my manual inputs above
    data.origin_country = data.origin_country.str.strip()


    # In[66]:


    # impact of assumptions

    print("The assumptions removed {}% of all null origin_country values".format(np.round((1 - data[data.origin_country == '0'].shape[0] / total_null_origins) * 100), 2))


    # ### Updating timezones & creating timedelta values

    # In[75]:


    # updating date fields to be in datetime format

    start = time.time()

    data.first_event_time = pd.to_datetime(data.first_event_time, infer_datetime_format=True)
    data.first_event_time = data.first_event_time.dt.tz_localize(None)

    data.imported_date = pd.to_datetime(data.imported_date, infer_datetime_format=True)

    end = time.time()

    print("The total time taken was {} seconds".format(end - start))


    # In[76]:


    # creating two dataframes for utc adjusted and local time first_event_time fields
    data_utctz = data[data.timezone_awareness == 'utc_time'].copy()

    data_localtz = data[data.timezone_awareness == 'location_local_time'].copy()


    # In[77]:


    # updating time format to be UTC for first_event_time + timezone_awareness = utc_time
    start = time.time()

    data_utctz.insert(loc=len(data_utctz.columns), column='first_event_time_utc', value=data_utctz.first_event_time)

    end = time.time()

    print("--- %s seconds ---" % np.round((end - start), 2))


    # In[ ]:

    local_missing_country = [country for country in data_localtz.origin_country.unique().tolist() if country not in COUNTRY_TIMEZONE_DICT]
    print("These are the countries missing from the dictionary above " + str(local_missing_country))

    # defining functions and dictionary
    start = time.time()

    # adding /utc updated first event time to data
    data_localtz = parallel_feature_calculation(data_localtz, apply_first_event_time_utc,
                                                num_processes=num_processes, num_partitions=num_partitions)

    end = time.time()
    print("--- Adding UTC timezone costs %s seconds ---" % np.round((end - start), 2))


    # In[ ]:


    # combining adjusted UTC & loacl timezone dataframes

    data = pd.concat((data_utctz, data_localtz), axis=0, sort=False)

    data = data.reset_index()

    # checking how many NA values are in the data
    try:
        a = data[data.first_event_time_utc == 'NA'].shape[0]
        print('There are {} errorenous values within the origin_country field for timezone corrections'.format(str(a)))
    except:
        print('No Nulls')


    # In[15]:


    # removing the NA values from the list
    data = data[data.first_event_time_utc != 'NA']


    # In[16]:


    # converting first event time utc field to datetime
    data.first_event_time_utc = pd.to_datetime(data.first_event_time_utc, infer_datetime_format=True, utc=True)

    # removing the UTC encoding for the first_event_time_date

    # data.first_event_time_utc = data.first_event_time_utc.dt.tz_localize(None)

    # creating new timedelta column in seconds
    data['timedelta_in_secs'] = (data.first_event_time_utc - data.imported_date).astype('timedelta64[s]')

    # creating new timedelta columns

    data['timedelta_in_mins'] = (data.first_event_time_utc - data.imported_date).astype('timedelta64[m]')
    data['timedelta_in_hrs'] = (data.first_event_time_utc - data.imported_date).astype('timedelta64[h]')


    # In[17]:


    # getting values greater than 0
    data = data[data.timedelta_in_secs > 0]


    # In[18]:


    # getting values less than 48 hours
    data = data[data.timedelta_in_mins <= 14400]


    # ### Encoding Fields to add catch all buckets

    # In[19]:


    # # encoding parcel_input_source
    start = time.time()

    data['parcel_input_source_count'] = data.groupby(['parcel_input_source'])['parcel_input_source'].transform('count')

    # creating encoding for org_carrier
    parcel_input_encoding = []
    for index, row in data[['parcel_input_source_count']].iterrows():
        a = row[0]
        b = data.shape[0]
        c = a / b
        if c < 0.05:
            parcel_input_encoding.append(index)

    # marking parcel_input_sources less than 5%
    data.loc[parcel_input_encoding, 'parcel_input_source'] = 'Other'


    end = time.time()
    print("--- Encode parcel_input_source_count %s seconds ---" % np.round((end - start), 2))


    # In[20]:


    # # encoding carrier_id
    start = time.time()

    data['carrier_count'] = data.groupby(['carrier_name'])['carrier_name'].transform('count')

    # creating encoding for org_carrier
    carrier_encoding = []
    for index, row in data[['carrier_count']].iterrows():
        a = row[0]
        b = data.shape[0]
        c = a / b
        if c < 0.005:
            carrier_encoding.append(index)

    data.loc[carrier_encoding, 'carrier_name'] = 'Other'


    end = time.time()
    print("--- Encode carrier_count %s seconds ---" % np.round((end - start), 2))


    # In[22]:


    # encoding org_carrier logic
    start = time.time()

    # creating org_carrier field
    data['org_carrier'] = data['org_name'].map(str) + "_" + data['carrier_name']

    # creating value counts column - org_carreir
    data['org_carrier_count'] = data.groupby(['org_name', 'carrier_name'])['org_name'].transform('count')

    # creating encoding for org_carrier
    org_carrier_encoding = []
    for index, row in data[['org_carrier_count']].iterrows():
        a = row[0]
        b = data.shape[0]
        c = a / b
        if c < 0.005:
            org_carrier_encoding.append(index)

    # marking org_carriers less than 10%
    data.loc[org_carrier_encoding, 'org_carrier'] = 'Other'


    end = time.time()
    print("--- Encode org_carrier_count %s seconds ---" % np.round((end - start), 2))


    # In[23]:


    # encoding origin_country logic

    start = time.time()

    # creating value counts for origin country
    data['origin_counts'] = data.groupby(['origin_country'])['origin_country'].transform('count')

    # creating encoding for origin_counts
    origin_encoding = []
    for index, row in data[['origin_counts']].iterrows():
        a = row[0]
        b = data.shape[0]
        c = a / b
        if c < .0005:
            origin_encoding.append(index)

    data.loc[origin_encoding, 'origin_country'] = 'Other'



    end = time.time()
    print("--- Encode origin_country %s seconds ---" % np.round((end - start), 2))


    # In[24]:


    # encoding org_name logic

    start = time.time()

    # creating value counts for origin country
    data['org_counts'] = data.groupby(['org_name'])['org_name'].transform('count')

    # creating encoding for origin_counts
    org_name_encoding = []
    for index, row in data[['org_counts']].iterrows():
        a = row[0]
        b = data.shape[0]
        c = a / b
        if c < .0005:
            org_name_encoding.append(index)

    data.loc[org_name_encoding, 'org_name'] = 'Other'



    end = time.time()
    print("--- Encode org_counts %s seconds ---" % np.round((end - start), 2))


    # ### Creating Label value

    # In[25]:


    data = data.reset_index()


    # #### please note the parts of code referring to Label are only for training - the Label value can also be changed during training
    #

    # In[26]:


    start = time.time()

    label_list = []

    for index, row in data[['timedelta_in_mins']].iterrows():
        if row['timedelta_in_mins'] <= 10:
            label_list.append(0)
        elif row['timedelta_in_mins'] <= 20:
            label_list.append(1)
        elif row['timedelta_in_mins'] <= 40:
            label_list.append(2)
        elif row['timedelta_in_mins'] <= 60:
            label_list.append(3)
        elif row['timedelta_in_mins'] <= 90:
            label_list.append(4)
        elif row['timedelta_in_mins'] <= 120:
            label_list.append(5)
        elif row['timedelta_in_mins'] <= 180:
            label_list.append(6)
        elif row['timedelta_in_mins'] <= 240:
            label_list.append(7)
        elif row['timedelta_in_mins'] <= 300:
            label_list.append(8)
        elif row['timedelta_in_mins'] <= 360:
            label_list.append(9)
        elif row['timedelta_in_mins'] <= 420:
            label_list.append(10)
        elif row['timedelta_in_mins'] <= 480:
            label_list.append(11)
        elif row['timedelta_in_mins'] <= 600:
            label_list.append(12)
        elif row['timedelta_in_mins'] <= 900:
            label_list.append(13)
        elif row['timedelta_in_mins'] <= 1320:
            label_list.append(14)
        elif row['timedelta_in_mins'] <= 1800:
            label_list.append(15)
        elif row['timedelta_in_mins'] <= 2880:
            label_list.append(16)
        elif row['timedelta_in_mins'] <= 4080:
            label_list.append(17)
        elif row['timedelta_in_mins'] <= 5640:
            label_list.append(18)
        elif row['timedelta_in_mins'] <= 7260:
            label_list.append(19)
        elif row['timedelta_in_mins'] <= 9660:
            label_list.append(20)
        elif row['timedelta_in_mins'] <= 12180:
            label_list.append(21)
        elif row['timedelta_in_mins'] <= 14400:
            label_list.append(22)
        else:
            label_list.append("NA")

    data['Label'] = label_list

    end = time.time()
    print("--- %s seconds ---" % np.round((end - start), 2))


    # ### Saving data file before removing headers

    # In[28]:
    if should_upload_to_s3 is True:
        data_file_name = file_name or "preprocessed_data_{}".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))
        # data.to_csv(data_file_name)

        # In[2]:


        # saving data file to S3
        upload_data_to_s3(data=data, input_file_name=data_file_name)

        print('Uploaded data to s3: {}'.format(data_file_name))

    # In[5]:


    # Dropping 'org_carrier_count' and 'org_carrier'
    data.drop(['pk', 'parcel_id', 'org_id', 'org_slug', 'carrier_id', 'parcel_product', 'imported_date',
               'first_event_time', 'timedelta_in_seconds', 'first_event_time_utc', 'timedelta_in_secs',
               'timedelta_in_hrs', 'origin_counts', 'timezone_awareness',
               'org_carrier_count', 'origin_counts', 'carrier_count', 'parcel_input_source_count', 'org_counts', 'org_carrier'], axis=1, inplace=True)


    # In[6]:


    # creating Y variable
    y = data['Label']
    y = y.values.astype('int64')


    # In[9]:


    # creating X variable

    # data = data.drop('Label',axis=1).replace(np.nan, 0, regex=True).astype('object')

    X = pd.get_dummies(data, drop_first=True)

    X = X.apply(pd.to_numeric)

    # adjusting dataframe and making non_numerical references
    pd.set_option('precision', 0)

    # column_names = pd.DataFrame(X.columns)
    X = X.values.astype(float)
    return X, y

# ### Usage

# data, data_orig = import_data()
# X, y = preprocess_data(data)

# In[ ]:


# upload_data_to_s3(data=X, input_file_name='MLR_X.csv')

# In[24]:

# upload_data_to_s3(data=y, input_file_name='MLR_y.csv')

# X = input for AWS Endpoint

# y = Labels for training
