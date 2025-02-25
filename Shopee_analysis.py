#!/usr/bin/env python
# coding: utf-8

# ## Name: Aidil Adham bin Ismail
# ## Position: Junior Data Scientist
# ## Task: Technical Test - eBdesk Malaysia Sdn Bhd

# # 

# ### Task Description
# 
# ### 1. The Shopee data was collected daily during the mentioned time range. Based on thecurrent crawling system, some products may be crawled in several days. We want to get listed products from the data that was listed during May 2023.

# In[1375]:


# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[1376]:


# Import Dataset

df = pd.read_csv('20240121_shopee_sample_data_ (1).csv')

# Convert w_date to datetime format and filter for May 2023
df = df[pd.to_datetime(df['w_date'], errors='coerce').between('2023-05-01', '2023-05-31')]

df.head()


# In[1377]:


# Check Total Number of Rows or Columns
df.shape


# In[1378]:


# Check columns
df.columns


# In[1379]:


# Keep Relevant column for task and remove unnecessary column

df = df[['price_ori','item_category_detail','specification',
         'w_date','price_actual','total_rating','total_sold',
         'favorite']]

df.head()


# In[1380]:


# Check Total Number of Rows or Columns
df.shape


# In[1381]:


# Convert 'K' to numeric
def convert_k(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^0-9.kK]', '', regex=True)  # Keep only numbers and 'K'
                         .str.replace('K', 'e3', case=False), errors='coerce')  # Convert 'K' to 'e3' (1000)

# Apply function to selected columns
columns = ['favorite', 'total_rating', 'total_sold']
df[columns] = df[columns].apply(convert_k)

df.head()


# #### Treating Missing Values

# In[1382]:


# Check Missing values
df.isnull().sum()


# In[1383]:


df.dtypes


# In[1384]:


# Convert Numeric value into float or int

numeric_cols = ['price_ori', 'price_actual']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# In[1385]:


df.dtypes


# In[1386]:


# Fill in missing values in price_ori, price_actual, total_rating, total_sold, favorite using MEDIAN
df['price_ori'] = df['price_ori'].fillna(df['price_ori'].median())

df['price_actual'] = df['price_actual'].fillna(df['price_actual'].median())

df['total_rating'] = df['total_rating'].fillna(df['total_rating'].median())

df['total_sold'] = df['total_sold'].fillna(df['total_sold'].median())

df['favorite'] = df['favorite'].fillna(df['favorite'].median())

#Fill in missing values in specification using MODE
df['specification'] = df['specification'].fillna(df['specification'].mode()[0])


df.head()


# In[1387]:


# Check Missing values
df.isnull().sum()


# #### Treating Duplicated Rows

# In[1388]:


#Check Duplicated Rows
df[df.duplicated(keep=False)]


# In[1389]:


# Remove duplicates and keep the first
df.drop_duplicates(keep='first', inplace = True) #Inplace true mean no need to assigned into new variable

# Recheck it again
df[df.duplicated(keep=False)]


# In[1390]:


df.shape


# In[1391]:


# Dataset below show the data that was listed during May 2023.

df.head(10)


# #### Dataset have been cleaned, now lets jump to next question

# ### Univariate Analysis
# ### 2. Show how many products are crawled each date.

# In[1392]:


# Count the number of products for each date
product_crawled = df['w_date'].value_counts().sort_index()

product_crawled


# In[1417]:


# Bar chart to show Number of products crawled per date

plt.figure(figsize=(15, 7))
ax = sns.countplot(x='w_date', data=df, order=sorted(df['w_date'].unique()))
for label in ax.containers:
    ax.bar_label(label)
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Number of Products")
plt.title("Number of Products Crawled Each Day")

plt.show()


# ##### Conclusion
# 
# 1. Some days had way more products crawled, like May 1st (3,546) and May 13th (3,645). Maybe it was a scheduled run or a big update.
# 
# 2. Most days had around 500-1,000 products, which looks like a normal daily crawl.
# 
# 3. May 12th (2,436) had a suddenly jump, maybe something special happened that day.
# 
# 4. May 4th was very low (200), so maybe there was an issue or less data available.

# ### 3. Show number of listing product based on location. You can extract this from “specification” field.

# In[1394]:


# States in Malaysia
states = ["Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang",
          "Perak", "Perlis", "Pulau Pinang", "Sabah", "Sarawak", "Selangor",
          "Terengganu", "Kuala Lumpur", "Putrajaya", "Labuan"]

# Extract the state from 'specification'
def get_location(text):
    for state in states:
        if isinstance(text, str) and state in text:
            return state
    return "Oversea"

# Apply function and count
df["location"] = df["specification"].apply(get_location)
df["location"].value_counts()


# In[1395]:


# Bar Chart to show Number of Listings by State

plt.figure(figsize=(12, 6))

ax = sns.countplot(x=df["location"], order=df["location"].value_counts().index)
for label in ax.containers:
    ax.bar_label(label)

plt.xticks(rotation=45)
plt.xlabel("Location")
plt.ylabel("Number of Listings")
plt.title("Number of Listings by Location")

plt.show()


# In[1396]:


# Pie chart to show proportion of listing by Location

plt.figure(figsize=(8, 8))
df["location"].value_counts().plot(kind="pie", autopct="%.0f%%")
plt.title("Proportion of Listings by Location")
plt.legend(labels=df['location'].value_counts().index, loc="upper right", bbox_to_anchor=(1.3, 1))

plt.show()


# ###### Conclusion
# - The highest number of listings (35%) comes from Selangor, showing its dominance in the market.
# - Overseas listings (21%) indicate a significant international supply.
# - KL (14%) and Johor (7%) are also key hubs for listings.
# - Perlis, Sarawak, Sabah, and Putrajaya have minimal listings, likely due to lower demand or logistical challenges.
# - Increasing listings in underserved states, especially in East Malaysia, could improve accessibility and business reach.

# ### 4. Item category detail may have this format: “Shopee | Women's Clothing | Outerwear | Coats & Jackets”.
# We can split it into:
# 
# Main category : Women’s Clothing
# 
# Subcategory 1 : Outerwear
# 
# Subcategory 2 : Coats & Jackets
# 
# a. Show number of listing products based on main category.
# 
# b. For the top 3 main categories, show the top 5 subcategory 1 for that main category based on number of products

# In[1397]:


# Split by '|' and strip spaces and remove 'Shopee'
df[['main_category', 'subcategory_1', 'subcategory_2']] = (
    df['item_category_detail'].str.split('|', expand=True).iloc[:, 1:4].apply(lambda x: x.str.strip()))


# In[1398]:


# a. Show number of listing products based on main category.

df['main_category'].value_counts()


# In[1399]:


# Bar chart to show Number of listings per main category

plt.figure(figsize=(12, 6))

ax = sns.countplot(x=df["main_category"], order=df["main_category"].value_counts().index)
for label in ax.containers:
    ax.bar_label(label)

plt.xticks(rotation=90)
plt.xlabel("Main Category")
plt.ylabel("Number of Listings")
plt.title("Number of Listings per Main Category")
plt.show()


# ###### Conclusion
# - The most listed product category is Men Clothes with 1,904 listings, followed by Health & Beauty (1,868) and Women Clothes (1,691).
# - These three categories dominate the marketplace, indicating high demand and product availability.
# - Mobile & Accessories and Baby & Toys also have a significant presence, suggesting strong consumer interest in these categories.
# - Categories like Gaming & Consoles, Tickets & Vouchers, and Others have the fewest listings, showing relatively low product diversity or demand.
# - This distribution suggests a focus on fashion, personal care, and daily-use items, reflecting consumer preferences in the marketplace.

# In[1400]:


plt.figure(figsize=(10, 10))
df['main_category'].value_counts().plot.pie(autopct='%.0f%%')
plt.title("Proportion of Listings per Main Category")
plt.legend(labels=df['main_category'].value_counts().index, loc="upper right", bbox_to_anchor=(1.3, 1))

plt.show()


# ###### Conclusion
# - Men Clothes (12%), Health & Beauty (11%), and Women Clothes (10%) are the top three categories, making up a significant portion of total listings.
# - Mobile & Accessories (9%) and Baby & Toys (8%) also hold a strong presence, indicating their popularity in the marketplace.
# - Categories like Home & Living (6%), Groceries & Pets (5%), and Home Appliances (4%) contribute moderately to the total listings.
# - Less dominant categories, such as Gaming & Consoles, Tickets & Vouchers, and Travel & Luggage, make up only 1% or less, indicating lower product availability or demand.
# - The overall distribution highlights a strong focus on fashion, personal care, and everyday essentials, reflecting consumer buying trends.

# In[1401]:


# b. For the top 3 main categories, show the top 5 subcategory 1 for that main category based on number of products

top_3_main = df['main_category'].value_counts().index[:3]

for category in top_3_main:
    print(f"\nTop 5 subcategories for {category}:")
    print(df[df['main_category'] == category]['subcategory_1'].value_counts().head(5))


# In[1402]:


# Create an empty list to store subcategory data
subcategory_data = []

# Store the top 5 subcategories for each main category
for category in top_3_main:
    subcategory_counts = df[df['main_category'] == category]['subcategory_1'].value_counts().head(5)
    for subcategory, count in subcategory_counts.items():
        # Rename 'Others' to distinguish between categories
        if subcategory == "Others":
            subcategory = f"{category} - Others"
        subcategory_data.append({"Main Category": category, "Subcategory": subcategory, "Count": count})

# Convert to DataFrame
subcategory_df = pd.DataFrame(subcategory_data)


# In[1403]:


# Plot using Seaborn
plt.figure(figsize=(14, 7))

ax = sns.barplot(data=subcategory_df, x="Subcategory", y="Count", hue="Main Category", dodge=False)
for label in ax.containers:
    ax.bar_label(label)

plt.legend(title="Main Category", fontsize=10)
plt.xlabel("Subcategory", fontsize=12)
plt.ylabel("Number of Listings", fontsize=12)
plt.title("Top 5 Subcategory 1 for the Top 3 Main Categories", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.grid(axis='y', linestyle="--", alpha=0.7)

plt.show()


# ###### Conclusion
# 1. Men Clothes:
# - The most listed subcategory is Sets (290 listings), significantly leading over other subcategories.
# - Suits (194 listings) and Shirts (152 listings) indicate strong demand for formal and casual attire.
# - Others (170 listings) suggests a variety of miscellaneous clothing items are available.
# - Pants (149 listings) show stable demand compared to shirts.
# 
# 2. Health & Beauty:
# - Foot Care (123 listings) and Sun Care (121 listings) are the most listed subcategories, highlighting strong interest in skincare and self-care products.
# - Skincare (115 listings), Pedicure & Manicure (115 listings), and Eye Makeup (114 listings) indicate a balanced demand for beauty and cosmetic products.
# 
# 3. Women Clothes:
# - Sports & Beachwear (185 listings) and Outerwear (181 listings) dominate the listings, showing high demand for active and seasonal wear.
# - Tops (169 listings) and Playsuits & Jumpsuits (136 listings) suggest trendy fashion items are popular.
# - Others (150 listings) indicate diverse clothing styles beyond specific categories.

# ### 5. Show price range for each main category

# In[1404]:


df['price_actual'] = pd.to_numeric(df['price_actual'], errors='coerce')

# Remove outliers using IQR method
Q1 = df['price_actual'].quantile(0.25)
Q3 = df['price_actual'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['price_actual'] >= (Q1 - 1.5 * IQR)) & (df['price_actual'] <= (Q3 + 1.5 * IQR))]


# In[1405]:


price_range = df.groupby('main_category')['price_actual'].agg(['min', 'max']).reset_index()
price_range


# ###### Conclusion
# - The minimum price for most categories starts from 0.00 to 0.35, indicating the presence of very low-cost or promotional items.
# - The maximum price varies across categories, with the highest prices reaching above 190 in categories like Automotive, Baby & Toys, Cameras & Drones, and Home Appliances.
# - Travel & Luggage has a relatively lower maximum price (34.90) compared to other categories.

# In[1406]:


# Box Plot to show price distribution per category

plt.figure(figsize=(12, 6))
sns.boxplot(x='main_category', y='price_actual', data=df)
plt.xticks(rotation=90)
plt.xlabel("Main Category")
plt.ylabel("Price")
plt.title("Price Distribution for Each Main Category")
plt.show()


# ###### Conclusion
# - Some categories have a wide range of prices, meaning they contain both low-cost and expensive products.
# - There are outliers in many categories, showing that a few products have significantly higher prices compared to the majority.

# In[1407]:


#Histogram to see overall price distribution

plt.figure(figsize=(10, 5))
plt.hist(df['price_actual'], bins=30, edgecolor='black')
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Overall Price Distribution")
plt.show()


# ###### Conclusion
# - The majority of products are priced under 25, as seen from the high frequency of low prices.
# - There are fewer expensive products, meaning most items in this dataset are affordable.

# ### Multivariate Analysis
# ### 6. Show the revenue for each main category in descending order

# In[1408]:


# Calculate The Total Revenue

df['revenue'] = df['price_actual'] * df['total_sold']

df['revenue']


# In[1409]:


# Total Revenue for Each Main Category

revenue_per_category = df.groupby('main_category')['revenue'].sum().reset_index()
revenue_per_category


# In[1410]:


# Bar Chart to show Revenue Per Category

plt.figure(figsize=(12, 6))
sns.barplot(x='revenue', y='main_category', data=revenue_per_category.sort_values(by='revenue', ascending=False))
plt.xlabel("Total Revenue")
plt.ylabel("Main Category")
plt.title("Total Revenue by Main Category")
plt.show()


# ###### Conclusion
# - The highest revenue comes from Health & Beauty, Home & Living, and Mobile & Accessories, indicating strong customer demand in these categories.
# - On the other hand, Tickets & Vouchers and Travel & Luggage contribute the least revenue.

# In[1411]:


#Scatter Plot Price vs Total Sold

plt.scatter(df['price_actual'], df['total_sold'])
plt.ylabel("Price")
plt.xlabel("Total Sold")
plt.title("Price vs. Total Sold")
plt.show()


# ###### Conclusion
# - The scatter plot indicates an inverse relationship between price and total sales.
# - Lower-priced products tend to sell more units, making total sales volume a crucial factor in revenue generation.

# In[1412]:


#CORRELATION COEFICIENT

revenue_corr = df[['price_actual', 'total_sold', 'revenue']].corr(numeric_only=True).round(2)

revenue_corr


# In[1413]:


# Correlation Heatmap between Revenue and Other Variables

plt.figure(figsize=(5,5))
sns.heatmap(revenue_corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Revenue and Other Variables")
plt.show()


# ###### Conclusion
# - The heatmap shows that total_sold has a stronger correlation (0.53) with revenue compared to price_actual (0.18).
# - This suggests that the number of items sold has a bigger impact on total revenue than product price.
