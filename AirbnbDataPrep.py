####################################################################################
# Explainable AI - Airbnb Booking Rate Model
# data preparation module
# sean x wang
####################################################################################
import pandas as pd
import numpy as np
import datetime
import s3fs

# calculate listing booking activity from calendar data to used as training target
def getBookingScore(calendarB_url, calendarA_url, duration=30):    
    calendarB = pd.read_csv(calendarB_url,index_col=0,parse_dates=["date"],compression='gzip')

    # define target time period
    first = calendarB['date'].min()
    last = first + datetime.timedelta(days=duration)

    # select records by time range
    calendarB_target = calendarB[calendarB['date'].between(first, last, inclusive=True)]

    # unstack flattens t/f value per listing
    availabilityB = calendarB_target.groupby('listing_id')['available'].value_counts().unstack().fillna(0)

    # ratio of booking
    bookingB = availabilityB
    bookingB['ratio'] = availabilityB.f / (availabilityB.f + availabilityB.t)

    calendarA = pd.read_csv(calendarA_url,index_col=0,parse_dates=["date"],compression='gzip')

    # select records by time range
    calendarA_target = calendarA[calendarA['date'].between(first, last, inclusive=True)]

    # unstack flattens t/f value per listing
    availabilityA = calendarA_target.groupby('listing_id')['available'].value_counts().unstack().fillna(0)

    # ratio of booking
    bookingA = availabilityA
    bookingA['ratio'] = availabilityA.f / (availabilityA.f + availabilityA.t)

    # listing that has no change or an increase in booking rate
    A2B = pd.merge(bookingA, bookingB, how='right', on='listing_id')

    #if calender data only shows up for B, assume point A booking starts at 0
    A2B.ratio_x = A2B.ratio_x.fillna(0)

    # listing that always has nothing available A to B should be dropped
    #if time A ratio_x is 1, and time B ratio_y is still 1, drop them since they are stale unbookable listings that screw the model 
    A2B = A2B[~(A2B.ratio_x==1.0) | ~(A2B.ratio_y==1.0)]

    # since we already dropped 1/1 pair, then we are left with 1/<1 pair
    #if time A ratio_x is 1 and time B ratio_y changed to <1, change time A ratio to 0 (because likely host updated availability after A)
    A2B.ratio_x = A2B['ratio_x'].map(lambda x: 0 if x==1.0 else x)

    A2B['booking_score'] = A2B.ratio_y - A2B.ratio_x

    # drop outlier with negative booking increase
    A2B = A2B[ A2B.booking_score >= 0 ]

    return A2B

# strip leading $ and convert to integer cost
def convertCost(price):
    price = price.str[1:-3]
    price = price.str.replace(",", "")
    price = price.astype('float32')
    return price

# get listing features from raw data
def getListingFeatures(listings_url, date):
    raw_df = pd.read_csv(listings_url,compression='gzip',index_col=0)
    print(f"The dataset contains {len(raw_df)} Airbnb listings")

    #drop features not used
    cols_to_drop = ['listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_name', 'host_location', 'host_about', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_verifications', 'calendar_last_scraped']
    df = raw_df.drop(cols_to_drop, axis=1)
    df.drop(['host_acceptance_rate', 'neighbourhood_group_cleansed'], axis=1, inplace=True)
    # keep host_listings_count only
    df.drop(['host_total_listings_count'], axis=1, inplace=True)
    df.drop(['neighbourhood', 'city', 'state', 'market', 'smart_location', 'country_code', 'country'], axis=1, inplace=True)
    df.drop(['has_availability'], axis=1, inplace=True)
    df.drop('experiences_offered', axis=1, inplace=True)
    df.drop('street', axis=1, inplace=True)

    df.drop(['zipcode'], axis=1, inplace=True)
    df.drop(['amenities'], axis=1, inplace=True)

    # Replacing columns with f/t with 0/1
    df.replace({'f': 0, 't': 1}, inplace=True)

    # process category and convert to number 
    response_cats = df['host_response_time'].unique()
    # replace each category value with its number
    for i in range(len(response_cats)):
        if response_cats[i] == 'within an hour':
            to_replace = 1
            df['host_response_time'] = df['host_response_time'].replace(response_cats[i], to_replace)
        elif response_cats[i] == 'within a few hours':
            to_replace = 5
            df['host_response_time'] = df['host_response_time'].replace(response_cats[i], to_replace)
        elif response_cats[i] == 'within a day':
            to_replace = 20
            df['host_response_time'] = df['host_response_time'].replace(response_cats[i], to_replace)
        elif response_cats[i] == 'a few days or more':
            to_replace = 100
            df['host_response_time'] = df['host_response_time'].replace(response_cats[i], to_replace)
        else:
            print('do nothing for',response_cats[i])

    # Removing the % sign from the host_response_rate string and converting to an integer
    df.host_response_rate = df.host_response_rate.str[:-1].astype('float64')

    # convert cost to integer
    df.price = convertCost(df.price)
    df.weekly_price = convertCost(df.weekly_price)
    df.monthly_price = convertCost(df.monthly_price)

    df.security_deposit = convertCost(df.security_deposit)
    df.security_deposit.fillna(0, inplace=True)
    df.cleaning_fee = convertCost(df.cleaning_fee)
    df.cleaning_fee.fillna(0, inplace=True)
    df.extra_people = convertCost(df.extra_people)
    df.extra_people.fillna(0, inplace=True)

    # process category and convert to number of days
    df['calendar_updated'] = df['calendar_updated'].str.strip(" ago")
    calendar_cats = df['calendar_updated'].unique()
    # replace each category value with its number
    for i in range(len(calendar_cats)):
        if calendar_cats[i] == 'never':
            to_replace = 365 * 10
        elif calendar_cats[i] == 'today':
            to_replace = 0
        elif calendar_cats[i] == 'yesterday':
            to_replace = 1
        elif calendar_cats[i] == 'week':
            to_replace = 7
        else:
            multiple, period = calendar_cats[i].split(" ")
            if period == 'weeks':
                to_replace = int(multiple) * 7
            elif period == 'months':
                to_replace = int(multiple) * 30
            elif period == 'days':
                to_replace = int(multiple)
        df['calendar_updated'] = \
        df['calendar_updated'].replace(calendar_cats[i], to_replace)

    # Converting to datetime
    df.host_since = pd.to_datetime(df.host_since)
    df.first_review = pd.to_datetime(df.first_review)
    df.last_review = pd.to_datetime(df.last_review) # Converting to datetime
    
    # date features
    df['days_as_host'] = (date - df['host_since']).dt.days
    df['days_since_first_review'] = (date - df['first_review']).dt.days
    df['days_since_last_review'] = (date - df['last_review']).dt.days

    df.drop(['host_since','first_review', 'last_review'], axis=1, inplace=True)
    df.drop(['amenities', 'license'], axis=1, inplace=True)

    return df

# put feature and target together
def getBookingData(calendarB_url, calendarA_url, listings_url, date=pd.Timestamp.today()):
    A2B = getBookingScore(calendarB_url, calendarA_url)
    listings = getListingFeatures(listings_url, date)
    listings.drop(['availability_30', 'availability_60',  'availability_90', 'availability_365'], axis=1, inplace=True)
    
    A2B.index.names = ['id']
    df = pd.merge(A2B[['booking_score']], listings, how='left', left_on='id', right_on="id")

    return df