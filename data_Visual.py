import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import psycopg2 # Import for PostgreSQL connection
from sqlalchemy import create_engine # Import for pandas.read_sql_query
import os # To retrieve environment variables for credentials (recommended)
import json

# Load secrets from secrets.json
with open('secrets.json') as f:
    secrets = json.load(f)


# --- Streamlit Page Configuration ---
st.set_page_config(layout='wide', page_title="News Sentiment Analysis Dashboard")

# Set Streamlit theme options
st.config.set_option('theme.base', 'light')
st.config.set_option('theme.primaryColor', '#4CAF50')
st.config.set_option('theme.backgroundColor', '#F0F2F6')
st.config.set_option('theme.secondaryBackgroundColor', '#FFFFFF')
st.config.set_option('theme.textColor', '#020412')


# --- Database Configuration Parameters ---
# IMPORTANT: In a production environment, use environment variables or a secrets manager
# for these credentials, DO NOT hardcode them directly in the script.
DB_HOST = os.getenv("DB_HOST", "mypostgresdb.cze6w8e28cpt.ap-south-1.rds.amazonaws.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = secrets['db_password']
DB_PORT = os.getenv("DB_PORT", "5432")           # Default PostgreSQL port

DB_PROCESSED_DATA_TABLE_NAME = "final_data" # Table name where your processed data is stored


# --- Function to load data from PostgreSQL DB ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid repeated DB queries
def load_data_from_db():
    """
    Loads processed news data from the PostgreSQL database.
    Includes a check for required columns.
    """
    try:
        # Create a SQLAlchemy engine for pandas.read_sql_query
        engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

        # SQL query to select all data from the processed news table
        query = f"SELECT * FROM {DB_PROCESSED_DATA_TABLE_NAME};"

        # Read data into a pandas DataFrame
        df_from_db = pd.read_sql_query(query, engine)

        # Ensure 'date' column is datetime type after loading
        df_from_db['date'] = pd.to_datetime(df_from_db['date'])

        # --- Column Existence Check ---
        required_cols = ['date', 'category', 'cluster', 'compound', 'headline', 'content']
        for col in required_cols:
            if col not in df_from_db.columns:
                st.error(f"Error: Required column '{col}' not found in the database table '{DB_PROCESSED_DATA_TABLE_NAME}'. Please ensure your database table contains all necessary columns.")
                st.stop() # Stop the app if a critical column is missing

        st.success(f"Data loaded successfully from AWS PostgreSQL table.")
        return df_from_db

    except Exception as e:
        st.error(f"Error connecting to or loading data from PostgreSQL: {e}")
        st.stop() # Stop the app if data cannot be loaded


# --- Function to prepare data for heatmap and comparison table ---
@st.cache_data(ttl=3600)
def prepare_comparison_data(dataframe):
    """
    Calculates category counts and category-cluster distribution for comparison.
    """
    if dataframe.empty or 'category' not in dataframe.columns or 'cluster' not in dataframe.columns:
        return pd.DataFrame()

    # Original Category Counts
    category_overall_counts = dataframe['category'].value_counts().reset_index()
    category_overall_counts.columns = ['Category', 'Category_Counts']

    # Category-Cluster Distribution Counts
    category_cluster_counts = dataframe.groupby(['category', 'cluster']).size().reset_index(name='Cluster_Counts')
    category_cluster_counts.rename(columns={'category': 'Category', 'cluster': 'Cluster'}, inplace=True)

    # Merge to get total category count alongside category-cluster counts
    comparison_df = pd.merge(category_cluster_counts, category_overall_counts, on='Category', how='left')

    # Sort primarily by Cluster_Counts in descending order for the table
    comparison_df_sorted = comparison_df.sort_values(by='Cluster_Counts', ascending=False)

    return comparison_df_sorted

# --- Helper Function for Word Cloud Generation ---
@st.cache_data(ttl=3600)
def generate_word_cloud(text_series):
    """Generates a word cloud from a pandas Series of text."""
    if text_series.empty:
        return None # Return None if no text
    text = " ".join(text_series.astype(str).fillna(''))
    if not text.strip(): # Check if text is just whitespace
        return None
    return WordCloud(background_color='white').generate(text)


# --- Load the data at the start of the app ---
df = load_data_from_db()

# Add date-related columns needed for multiple plots once after loading data
if 'year' not in df.columns:
    df['year'] = df['date'].dt.year
if 'month' not in df.columns:
    df['month'] = df['date'].dt.month
if 'quarter' not in df.columns:
    df['quarter'] = df['date'].dt.quarter


# --- Sidebar Setup ---
st.sidebar.title("News Sentiment Analysis")

# --- Global Filters ---
st.sidebar.header("Filters")
min_overall_date = df['date'].min()
max_overall_date = df['date'].max()

global_start_date = st.sidebar.date_input("Start Date", min_value=min_overall_date, max_value=max_overall_date, value=min_overall_date, key="global_start_date")
global_end_date = st.sidebar.date_input("End Date", min_value=min_overall_date, max_value=max_overall_date, value=max_overall_date, key="global_end_date")

if global_start_date > global_end_date:
    st.sidebar.error("Global end date must be after start date.")
    # Set filtered_df to empty to prevent errors in visualizations
    filtered_df = pd.DataFrame()
else:
    filtered_df = df[(df['date'] >= pd.to_datetime(global_start_date)) & (df['date'] <= pd.to_datetime(global_end_date))]

all_categories = sorted(df['category'].unique().tolist())
selected_categories = st.sidebar.multiselect("Select Categories", all_categories, default=all_categories)

all_clusters = sorted(df['cluster'].unique().tolist())
selected_clusters = st.sidebar.multiselect("Select Clusters", all_clusters, default=all_clusters)

# Apply category and cluster filters
if not filtered_df.empty:
    filtered_df = filtered_df[
        (filtered_df['category'].isin(selected_categories)) &
        (filtered_df['cluster'].isin(selected_clusters))
    ]

if filtered_df.empty:
    st.info("No data available for the selected global filters. Please adjust your filter selections.")


st.sidebar.markdown("---")

# --- Analysis Option Checkboxes (Organized into Expanders) ---

# Clustering & Comparison
with st.sidebar.expander("Clustering & Comparison", expanded=True):
    comparison_table_checkbox = st.checkbox("View Category-Cluster Comparison Table", value=True)
    heatmap_checkbox = st.checkbox("Heatmap of Category vs. Cluster Counts", value=True)

# Sentiment Trends & Distribution
with st.sidebar.expander("Sentiment Trends & Distribution", expanded=True):
    sentiment_trend = st.checkbox("Overall Sentiment Trend by Category")
    year_month_trend = st.checkbox("Sentiment Trend (Year/Month-wise)")
    year_quarter_trend = st.checkbox("Sentiment Trend (Year/Quarter-wise)")
    bar_line_chart = st.checkbox("Category Sentiment (Bar and Line Chart)") # Renamed slightly
    category_month_year_comparison = st.checkbox("Category Sentiment (Month/Year Comparison)")
    category_sentiment_bar_chart = st.checkbox("Category-wise Sentiment Bar Chart")
    sentiment_distribution = st.checkbox("Sentiment Score Distribution")
    news_volume_trend = st.checkbox("News Volume Trend")
    sentiment_extremes_trend = st.checkbox("Sentiment Extremes Trend (Min/Max)") # New checkbox

# Word Clouds
with st.sidebar.expander("Word Clouds", expanded=True):
    word_cloud_by_cluster = st.checkbox("Word Cloud by Cluster")
    sentiment_word_cloud = st.checkbox("Sentiment Word Cloud")

# Article Details
with st.sidebar.expander("Article Details", expanded=True): # New expander
    top_n_articles = st.checkbox("Top N Articles by Sentiment") # New checkbox


st.sidebar.markdown("---")
st.sidebar.info("Select options from above to visualize news sentiment and clustering.")


# --- Main Content Area ---
st.title("Senticonomy")
st.write(" News Sentiment Analysis and Economic Impact Visualization .")

# Check if filtered_df is empty after global filters, if so, skip visualizations
if filtered_df.empty:
    st.stop() # Stop further execution of visualization code if no data


# --- Visualization Logic ---

# Word Cloud by Cluster
if word_cloud_by_cluster:
    st.header("Word Cloud by Cluster")
    st.write("Visualize the most frequent words in headlines for each identified cluster (based on global filters).")
    clusters = sorted(filtered_df['cluster'].unique())
    n_clusters = len(clusters)
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig_wc, axs_wc = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axs_wc = axs_wc.flatten() # Flatten for easy iteration if n_rows > 1

    for i, cluster in enumerate(clusters):
        if i < len(axs_wc):
            wc_img = generate_word_cloud(filtered_df[filtered_df['cluster'] == cluster]['headline'])
            if wc_img:
                axs_wc[i].imshow(wc_img, interpolation='bilinear')
                axs_wc[i].set_title(f"Cluster {cluster}")
            else:
                axs_wc[i].text(0.5, 0.5, "No text data for this cluster", horizontalalignment='center', verticalalignment='center', transform=axs_wc[i].transAxes)
                axs_wc[i].set_title(f"Cluster {cluster} (No Data)")
            axs_wc[i].axis('off')
    # Hide any unused subplots
    for i in range(n_clusters, len(axs_wc)):
        axs_wc[i].axis('off')
    plt.tight_layout()
    st.pyplot(fig_wc)
    plt.close(fig_wc)


# Overall Sentiment Trend by Category
if sentiment_trend:
    st.header("Overall Sentiment Trend by Category")
    st.write("Track the average sentiment score for each news category over time (based on global filters).")
    sentiment_trend_data = filtered_df.groupby(['year', 'category'])['compound'].mean().reset_index()
    if sentiment_trend_data.empty:
        st.info("No data for sentiment trend based on current filters.")
    else:
        fig = px.line(sentiment_trend_data, x='year', y='compound', color='category',
                      title='Sentiment Trend Over Time', markers=True,
                      labels={"compound": "Average Sentiment Score", "category": "News Category"})
        st.plotly_chart(fig)


# Sentiment Trend (Year/Month-wise)
if year_month_trend:
    st.header("Sentiment Trend (Year/Month-wise)")
    st.write("Compare category sentiment across selected years and months (based on global filters).")
    years_in_data = sorted(filtered_df['year'].unique().tolist())
    selected_years_month = st.multiselect("Select Years for Month Trend", years_in_data, default=list(years_in_data), key="month_trend_years")

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    selected_months_month = st.multiselect("Select Months for Month Trend", months, default=months, key="month_trend_months")

    fig = go.Figure()
    month_map = {month: i+1 for i, month in enumerate(months)}

    if not selected_years_month or not selected_months_month:
        st.info("Please select at least one year and one month to display the trend.")
    else:
        for year in selected_years_month:
            for month_name in selected_months_month:
                month_num = month_map[month_name]
                df_month_filtered = filtered_df[(filtered_df['year'] == year) & (filtered_df['month'] == month_num)]
                if not df_month_filtered.empty:
                    avg_compound = df_month_filtered.groupby(['category'])['compound'].mean().reset_index()
                    fig.add_trace(go.Scatter(x=avg_compound['category'], y=avg_compound['compound'],
                                             mode='lines', name=f"{year} - {month_name}"))

        fig.update_layout(title="Year/Month-wise Sentiment Trend", xaxis_title="Category", yaxis_title="Average Compound Sentiment Score")
        st.plotly_chart(fig)


# Sentiment Trend (Year/Quarter-wise)
if year_quarter_trend:
    st.header("Sentiment Trend (Year/Quarter-wise)")
    st.write("Compare category sentiment across selected years and quarters (based on global filters).")
    years_in_data_q = sorted(filtered_df['year'].unique().tolist())
    selected_years_quarter = st.multiselect("Select Years for Quarter Trend", years_in_data_q, default=list(years_in_data_q), key="quarter_trend_years")

    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    selected_quarters_quarter = st.multiselect("Select Quarters for Quarter Trend", quarters, default=quarters, key="quarter_trend_quarters")

    fig = go.Figure()

    if not selected_years_quarter or not selected_quarters_quarter:
        st.info("Please select at least one year and one quarter to display the trend.")
    else:
        for year in selected_years_quarter:
            for quarter_name in selected_quarters_quarter:
                quarter_num = int(quarter_name.split('Q')[1])
                df_quarter_filtered = filtered_df[(filtered_df['year'] == year) & (filtered_df['quarter'] == quarter_num)]
                if not df_quarter_filtered.empty:
                    avg_compound = df_quarter_filtered.groupby(['category'])['compound'].mean().reset_index()
                    fig.add_trace(go.Scatter(x=avg_compound['category'], y=avg_compound['compound'],
                                             mode='lines', name=f"{year} - {quarter_name}"))

        fig.update_layout(title="Year/Quarter-wise Sentiment Trend", xaxis_title="Category", yaxis_title="Average Compound Sentiment Score")
        st.plotly_chart(fig)


# Category Sentiment (Bar and Line Chart) - Now uses global filters
if bar_line_chart:
    st.header("Category Sentiment (Bar and Line Chart)")
    st.write("View average sentiment scores for categories within the globally selected date range.")
    if filtered_df.empty:
        st.info("No data available for the selected global date range to display this chart.")
    else:
        avg_compound = filtered_df.groupby(['category'])['compound'].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=avg_compound['category'], y=avg_compound['compound'], name='Average Sentiment (Bar)'))
        fig.add_trace(go.Scatter(x=avg_compound['category'], y=avg_compound['compound'], mode='lines+markers', name='Average Sentiment (Line)'))

        fig.update_layout(title=f"Category Sentiment within Global Filtered Range",
                          xaxis_title="Category",
                          yaxis_title="Average Sentiment Score")
        st.plotly_chart(fig)


# Category-Cluster Comparison Table
if comparison_table_checkbox:
    st.header("Category-Cluster Comparison Table")
    st.write("This table shows the distribution of original categories within the identified clusters (based on global filters).")
    if filtered_df.empty:
        st.info("No data available for the selected global filters to display the comparison table.")
    else:
        comparison_df_sorted = prepare_comparison_data(filtered_df)
        if comparison_df_sorted.empty:
            st.info("No data available for the selected global filters to display the comparison table.")
        else:
            st.dataframe(comparison_df_sorted, use_container_width=True)


# Category Sentiment (Month/Year Comparison)
if category_month_year_comparison:
    st.header("Category Sentiment (Month/Year Comparison)")
    st.write("Compare sentiment trends for specific categories across different months and years (based on global filters).")
    categories_in_data = sorted(filtered_df['category'].unique().tolist())
    selected_categories_comp = st.multiselect("Select Categories for Comparison", categories_in_data, default=list(categories_in_data), key="comp_categories")

    years_in_data_comp = sorted(filtered_df['year'].unique().tolist())
    selected_years_option_comp = st.multiselect("Select Years for Comparison", ['All Years'] + list(years_in_data_comp), default=['All Years'], key="comp_years")

    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                   7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    fig = go.Figure()

    if not selected_categories_comp or not selected_years_option_comp:
        st.info("Please select categories and years for comparison.")
    else:
        years_to_compare = list(years_in_data_comp) if 'All Years' in selected_years_option_comp else [y for y in selected_years_option_comp if y != 'All Years']

        if not years_to_compare:
            st.info("No valid years selected for comparison.")
        else:
            for category in selected_categories_comp:
                for year in years_to_compare:
                    df_comp_filtered = filtered_df[(filtered_df['category'] == category) & (filtered_df['year'] == year)]
                    if not df_comp_filtered.empty:
                        avg_compound_comp = df_comp_filtered.groupby(['month'])['compound'].mean().reset_index().sort_values(by='month')
                        fig.add_trace(go.Scatter(x=[month_names[m] for m in avg_compound_comp['month']], y=avg_compound_comp['compound'],
                                                 mode='lines+markers', name=f"{category} - {year}"))

            fig.update_layout(title="Category-wise Month and Year Sentiment Comparison",
                              xaxis_title="Month", yaxis_title="Average Compound Sentiment Score")
            st.plotly_chart(fig)


# Category-wise Sentiment Bar Chart
if category_sentiment_bar_chart:
    st.header("Category-wise Sentiment Bar Chart")
    st.write("Visualize the average sentiment score for each category across selected years (based on global filters).")
    categories_in_data_bar = sorted(filtered_df['category'].unique().tolist())
    years_in_data_bar = sorted(filtered_df['year'].unique().tolist())

    selected_years_bar = st.multiselect("Select Years for Bar Chart", years_in_data_bar, default=list(years_in_data_bar), key="bar_chart_years")

    fig = go.Figure()

    if not selected_years_bar:
        st.info("Please select at least one year for the bar chart.")
    else:
        for year in selected_years_bar:
            category_values = []
            for category in categories_in_data_bar:
                df_bar_filtered = filtered_df[(filtered_df['category'] == category) & (filtered_df['year'] == year)]
                if not df_bar_filtered.empty:
                    category_values.append(df_bar_filtered['compound'].mean().round(2))
                else:
                    category_values.append(0)
            # FIX: Convert year to string for the 'name' property
            fig.add_trace(go.Bar(x=categories_in_data_bar, y=category_values, name=str(year)))

        fig.update_layout(title="Category-wise Sentiment Bar Chart",
                          xaxis_title="Category", yaxis_title="Average Sentiment Score", barmode='group')
        st.plotly_chart(fig)


# Sentiment Word Cloud
if sentiment_word_cloud:
    st.header("Sentiment Word Cloud by Year")
    st.write("Explore words associated with positive, negative, and neutral sentiments for a selected year (based on global filters).")
    years_for_wc = sorted(filtered_df['year'].unique().tolist())
    if not years_for_wc:
        st.info("No data available for the selected global filters to generate sentiment word cloud.")
    else:
        selected_year_wc_sentiment = st.selectbox("Select Year for Sentiment Word Cloud", years_for_wc)

        df_wc_filtered = filtered_df[(filtered_df['year'] == selected_year_wc_sentiment)]

        positive_words = df_wc_filtered[df_wc_filtered['compound'] > 0.5]['content'].astype(str).fillna('')
        negative_words = df_wc_filtered[df_wc_filtered['compound'] < -0.5]['content'].astype(str).fillna('')
        neutral_words = df_wc_filtered[(df_wc_filtered['compound'] >= -0.5) & (df_wc_filtered['compound'] <= 0.5)]['content'].astype(str).fillna('')

        positive_wc = generate_word_cloud(positive_words)
        negative_wc = generate_word_cloud(negative_words)
        neutral_wc = generate_word_cloud(neutral_words)

        fig_sentiment_wc, ax_sentiment_wc = plt.subplots(1, 3, figsize=(20, 10))
        if positive_wc:
            ax_sentiment_wc[0].imshow(positive_wc, interpolation='bilinear')
        ax_sentiment_wc[0].set_title('Positive Sentiment')
        ax_sentiment_wc[0].axis('off')

        if negative_wc:
            ax_sentiment_wc[1].imshow(negative_wc, interpolation='bilinear')
        ax_sentiment_wc[1].set_title('Negative Sentiment') # Fixed: removed duplicate line
        ax_sentiment_wc[1].axis('off')

        if neutral_wc:
            ax_sentiment_wc[2].imshow(neutral_wc, interpolation='bilinear')
        ax_sentiment_wc[2].set_title('Neutral Sentiment')
        ax_sentiment_wc[2].axis('off')

        st.pyplot(fig_sentiment_wc)
        plt.close(fig_sentiment_wc)


# Heatmap of Category vs. Cluster Counts
if heatmap_checkbox:
    st.header("Heatmap of Category vs. Cluster Counts")
    st.write("A heatmap provides a dense overview of the counts for each Category-Cluster pair (based on global filters).")
    if filtered_df.empty:
        st.info("No data available for the selected global filters to display the heatmap.")
    else:
        heatmap_prep_df = prepare_comparison_data(filtered_df).copy()
        heatmap_prep_df['Cluster_Counts'] = heatmap_prep_df['Cluster_Counts'].fillna(0).astype(int)

        heatmap_data = heatmap_prep_df.pivot_table(
            index='Category',
            columns='Cluster',
            values='Cluster_Counts',
            fill_value=0
        )

        if heatmap_data.empty:
            st.info("No data available for the selected global filters to display the heatmap.")
        else:
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.0f',
                cmap='Blues',
                linewidths=.5,
                cbar_kws={'label': 'Count of Items'},
                ax=ax_heatmap
            )
            ax_heatmap.set_title(f'Heatmap of Category vs. Cluster Counts')
            ax_heatmap.set_xlabel('Cluster')
            ax_heatmap.set_ylabel('Original Category')
            ax_heatmap.tick_params(axis='x', rotation=0)
            ax_heatmap.tick_params(axis='y', rotation=0)
            plt.tight_layout()
            st.pyplot(fig_heatmap)
            plt.close(fig_heatmap)


# Sentiment Score Distribution
if sentiment_distribution:
    st.header("Sentiment Score Distribution")
    st.write("Visualize the distribution of sentiment scores by category or cluster (based on global filters).")

    group_by_option = st.selectbox("Group sentiment distribution by:", ["Category", "Cluster"])

    if filtered_df.empty:
        st.info("No data available for the selected global filters to display sentiment distribution.")
    else:
        group_col = 'category' if group_by_option == "Category" else 'cluster'

        if group_col not in filtered_df.columns:
            st.warning(f"Column '{group_col}' not found in the filtered data. Cannot display distribution.")
        else:
            fig_dist = px.histogram(filtered_df, x="compound", color=group_col,
                                    facet_col=group_col, facet_col_wrap=3,
                                    title=f"Distribution of Sentiment Scores by {group_by_option}",
                                    labels={"compound": "Sentiment Score", group_col: group_col},
                                    template="plotly_white",
                                    nbins=30
                                   )
            fig_dist.update_layout(height=600)
            fig_dist.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig_dist, use_container_width=True)


# News Volume Trend
if news_volume_trend:
    st.header("News Volume Trend")
    st.write("Analyze the daily or weekly volume of news articles (based on global filters).")

    aggregation_period = st.radio("Select aggregation period:", ("Daily", "Weekly"))

    if filtered_df.empty:
        st.info("No data available for the selected global filters to display news volume trend.")
    else:
        df_volume = filtered_df.copy()
        df_volume = df_volume.set_index('date')

        if aggregation_period == "Daily":
            volume_data = df_volume.resample('D').size().reset_index(name='Article Count')
            volume_data.columns = ['Date', 'Article Count']
            title = "Daily News Article Volume Over Time"
        else: # Weekly
            volume_data = df_volume.resample('W').size().reset_index(name='Article Count')
            volume_data.columns = ['Week', 'Article Count']
            title = "Weekly News Article Volume Over Time"

        if volume_data.empty:
            st.info(f"No data available for {aggregation_period} aggregation based on current filters.")
        else:
            fig_volume = px.line(volume_data, x=volume_data.columns[0], y='Article Count',
                                 title=title,
                                 labels={volume_data.columns[0]: "Date/Week", "Article Count": "Number of Articles"},
                                 template="plotly_white")
            st.plotly_chart(fig_volume, use_container_width=True)


# --- Sentiment Extremes Trend (Min/Max) ---
if sentiment_extremes_trend:
    st.header("Sentiment Extremes Trend (Min/Max)")
    st.write("Track the minimum (most negative) and maximum (most positive) sentiment scores over time (based on global filters).")

    if filtered_df.empty:
        st.info("No data available for the selected global filters to display sentiment extremes.")
    else:
        # Group by date and calculate min/max compound scores
        sentiment_extremes_data = filtered_df.groupby(filtered_df['date'].dt.date)['compound'].agg(['min', 'max', 'mean']).reset_index()
        sentiment_extremes_data.columns = ['Date', 'Min Sentiment', 'Max Sentiment', 'Mean Sentiment']

        fig_extremes = go.Figure()
        fig_extremes.add_trace(go.Scatter(x=sentiment_extremes_data['Date'], y=sentiment_extremes_data['Max Sentiment'],
                                         mode='lines', name='Max Positive Sentiment', line=dict(color='green')))
        fig_extremes.add_trace(go.Scatter(x=sentiment_extremes_data['Date'], y=sentiment_extremes_data['Min Sentiment'],
                                         mode='lines', name='Max Negative Sentiment', line=dict(color='red')))
        fig_extremes.add_trace(go.Scatter(x=sentiment_extremes_data['Date'], y=sentiment_extremes_data['Mean Sentiment'],
                                         mode='lines', name='Mean Sentiment', line=dict(color='blue', dash='dot')))

        fig_extremes.update_layout(title='Daily Sentiment Extremes and Mean Over Time',
                                   xaxis_title='Date',
                                   yaxis_title='Sentiment Score',
                                   hovermode="x unified")
        st.plotly_chart(fig_extremes, use_container_width=True)


# --- Top N Articles by Sentiment ---
if top_n_articles:
    st.header("Top N Articles by Sentiment")
    st.write("View the most positive or most negative articles based on current filters.")

    if filtered_df.empty:
        st.info("No data available for the selected global filters to display top articles.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            sentiment_type = st.radio("Select Sentiment Type", ("Most Positive", "Most Negative"))
        with col2:
            num_articles = st.number_input("Number of Articles to Display (N)", min_value=1, max_value=50, value=5, step=1)

        if sentiment_type == "Most Positive":
            top_articles = filtered_df.sort_values(by='compound', ascending=False).head(num_articles)
        else: # Most Negative
            top_articles = filtered_df.sort_values(by='compound', ascending=True).head(num_articles)

        if top_articles.empty:
            st.info(f"No {sentiment_type.lower()} articles found for the selected filters.")
        else:
            for idx, row in top_articles.iterrows():
                with st.expander(f"**{row['headline']}** (Score: {row['compound']:.2f}) - {row['date'].strftime('%Y-%m-%d')}"):
                    st.write(f"**Category:** {row['category']}")
                    st.write(f"**Cluster:** {row['cluster']}")
                    st.write("**Content:**")
                    st.write(row['content'])
                st.markdown("---") # Add a separator


st.markdown("---")
st.success("Dashboard Ready!")