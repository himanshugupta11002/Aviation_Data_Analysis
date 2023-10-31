#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from pywaffle import Waffle
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import warnings
warnings.simplefilter("ignore")


# In[4]:


def calculate_percentage_cross_tab_with_style(df, x):
    cross_tab = pd.crosstab(df[x], df['satisfaction'])
    percentage_cross_tab = cross_tab.apply(lambda row: row / row.sum() * 100, axis=1)
    rounded_percentage_cross_tab = percentage_cross_tab.round(2)
    styled_percentage_cross_tab = rounded_percentage_cross_tab.style.background_gradient(cmap='Blues')
    
    return styled_percentage_cross_tab


# In[5]:


def create_grouped_bar_chart(x, y, df, color1, color2):
    satisfaction_percentage = (
        df.groupby([x, y]).size() /
        df.groupby([x]).size()
    ).reset_index(name='Percentage').round(4)

    satisfaction_percentage['Percentage'] *= 100

    fig = px.bar(
        satisfaction_percentage,
        x=x,
        y='Percentage',
        color=y,
        barmode='group',
        title=f'{x} vs. {y}',
        labels={'Percentage': 'Percentage of Customers'},
        color_discrete_sequence=[color1, color2],  # Custom colors
    )
    fig.update_xaxes(title_text=x, tickfont=dict(size=12))
    fig.update_yaxes(title_text='Percentage', tickfont=dict(size=12))
    fig.update_layout(
        title_font=dict(size=18),
        legend_title=dict(text=y),
        legend_font=dict(size=12),
        legend=dict(orientation='h', x=0.5, y=1.15),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area background
    )
    fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')

    fig.show()


# In[6]:


def create_custom_bar_plot(df, x, y, color1, color2):
    # Calculate percentages
    percentages = (
        df.groupby([x, y])[y]
        .count()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum())
        .reset_index(name='Percentage')
    )

    # Generate the title
    title = f'{x} vs. {y}'

    # Create the bar plot
    fig = px.bar(
        percentages,
        x=x,
        y='Percentage',
        color=y,
        title=title,
        labels={'Percentage': 'Percentage of Total'},
        color_discrete_sequence=[color1, color2],  # Custom colors
    )
    fig.update_xaxes(categoryorder='array')  # Sort x-axis categories by percentage
    fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    fig.update_layout(
    title_font=dict(size=18),
    legend_title=dict(text=y),
    legend_font=dict(size=12),
    legend=dict(orientation='h', x=0.5, y=1.15))
    # Show the plot
    fig.show()    


# In[7]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Print the shape of the train and test DataFrames
print(Fore.LIGHTBLUE_EX +"Shape of train DataFrame:", train.shape)
print(Fore.LIGHTBLUE_EX +"Shape of test DataFrame:", test.shape)


# In[8]:


pd.set_option('display.max_columns', None)


# In[9]:


print(Fore.LIGHTBLUE_EX + "Head of train DataFrame :")
train.head()


# In[10]:


print(Fore.LIGHTBLUE_EX + "Head of test DataFrame:")
test.head()


# In[11]:


print(Fore.LIGHTBLUE_EX + "Info for train DataFrame:")
train.info()


# In[12]:


print(Fore.LIGHTBLUE_EX + "\nInfo for test DataFrame:")
test.info()


# In[13]:


train_NaN_count = train.isna().sum()
print(Fore.LIGHTBLUE_EX + "NaN count in train DataFrame:")
train_NaN_count


# In[14]:


test_NaN_count = test.isna().sum()
print(Fore.LIGHTBLUE_EX + "\nNaN count in test DataFrame:")
test_NaN_count


# In[15]:


print(Fore.LIGHTBLUE_EX + "Duplicate rows in train DataFrame:")
train_duplicates = train[train.duplicated()]
train_duplicates


# In[16]:


print(Fore.LIGHTBLUE_EX + "\nDuplicate rows in test DataFrame:")
test_duplicates = test[test.duplicated()]
test_duplicates


# In[17]:


print(Fore.LIGHTBLUE_EX + "Numerical description for train DdataFrame:")
train.describe().style.background_gradient(cmap="Blues")


# In[18]:


print(Fore.LIGHTBLUE_EX + "\nNumerical description for test DataFrame:")
test.describe().style.background_gradient(cmap = 'Blues')


# In[19]:


styles = [
    {
        'selector': 'th',
        'props': [
            ('font-size', '14px'),
            ('background-color', '#c9e1f1'),
            ('text-align', 'center'),
            ('padding', '8px'),
        ],
    },
    {
        'selector': 'td',
        'props': [
            ('font-size', '12px'),
            ('text-align', 'center'),
            ('padding', '8px'),
        ],
    },
]


# In[20]:


print(Fore.LIGHTBLUE_EX +"\nCategorical Description: for train DataFrame:")
train.describe(include="object").style.set_table_styles(styles)


# In[21]:


print(Fore.LIGHTBLUE_EX + "\nCategorical Description: for test DataFrame:")
test.describe(include="object").style.set_table_styles(styles)


# In[22]:


train_unique_counts = train.nunique()
print(Fore.LIGHTBLUE_EX + "Number of unique values in each column of the dataframe:")
train_unique_counts


# In[23]:


satisfaction_counts = train['satisfaction'].value_counts()

# Define custom colors suitable for airplane passengers
custom_colors = ["#08306B", "#BBFFFF"]  

# Define custom icons for frown and smile
icons = ['\U0001F641', '\U0001F604']  # Unicode emojis for frown and smile

# Create a bar chart with custom icons
fig = go.Figure()

# Add bars with icons
for i, (satisfaction, count) in enumerate(satisfaction_counts.items()):
    fig.add_trace(go.Bar(
        x=[satisfaction],
        y=[count],
        name=satisfaction,
        marker=dict(color=custom_colors[i]),
    ))
    
    fig.add_annotation(
        x=satisfaction,
        y=count,
        text=icons[i],
        showarrow=False,
        font=dict(size=35)
    )

fig.update_layout(
    title_text="Distribution of Customer Satisfaction",
    xaxis_title="Satisfaction",
    yaxis_title="Count",
    xaxis=dict(tickvals=[0, 1], ticktext=satisfaction_counts.index)  # Set custom x-axis tick values and labels
)

fig.show()


# In[24]:


Satisfaction_counts = train['satisfaction'].value_counts()

# Define a custom color palette resembling a sky
sky_colors = ['#1E90FF', '#98F5FF']

fig = px.pie(
    values=Satisfaction_counts.values,
    names=Satisfaction_counts.index,
    title="Percentage of passengers who satisfied or not",
    color_discrete_sequence=sky_colors  # Custom sky-themed colors
)

fig.update_traces(textinfo='percent+label', pull=[0.03, 0.02], textfont=dict(size=18))  # Increase font size

fig.update_layout(
    showlegend=True,
    title_font=dict(size=22),  # Increase title font size
    width=800,  # Adjust the width of the figure
    height=600  # Adjust the height of the figure
)

fig.show()


# In[25]:


gender_counts = train['Gender'].value_counts()

# Calculate the percentage of each gender category
gender_percentage = (gender_counts / len(train)) * 100

# Create a figure for the Waffle chart
fig = plt.figure( 
    FigureClass=Waffle,
    rows=5,  # rows of people
    figsize=(11, 6),
    values=gender_percentage,  # data as percentages
    labels=[f"Female ({gender_percentage['Female']:.2f}%)", f"Male ({gender_percentage['Male']:.2f}%)"],  # legend labels with percentages
    colors=["#FF82AB", "#1E90FE"],  # Custom colors (baby pink and baby blue)
    icons=['female', 'male'],  # Use 'female' and 'male' symbols available in pywaffle
    legend={'loc': 'lower center',
            'bbox_to_anchor': (0.5, -0.5),
            'ncol': len(gender_counts),
            'framealpha': 0,
            'fontsize': 20
            },
    icon_size=30,  # Size of icons (people)
    icon_legend=True,
    title={'label': 'Gender Distribution',
           'loc': 'center',
           'fontdict': {'fontsize': 20}
           }
)

plt.show()


# In[26]:


calculate_percentage_cross_tab_with_style(train, 'Gender')


# In[27]:


gender_satisfaction_count = train.groupby(["Gender", "satisfaction"]).size().reset_index(name="Count")

# Create two pie charts, one for each gender
fig = px.pie(
    gender_satisfaction_count,
    values="Count",
    names="satisfaction",
    title="Satisfaction Distribution by Gender",
    color="satisfaction",
    #color_discrete_sequence=['#98F5FF', '#08306B'],  # Define custom colors
    color_discrete_sequence=['#98F5FF', '#193EB0'],  # Define custom colors
    facet_col="Gender",
)

# Update layout for aesthetics
fig.update_traces(textinfo='percent+label', pull=[0.03,0])

# Show the plot
fig.show()


# In[28]:


train['Age'].unique()


# In[29]:


age_range_fig = px.histogram(train, x="Age", title="Age Distribution of Customers")

# Customize the color palette for the plot
age_range_fig.update_traces(marker=dict(color='skyblue'))

# Customize the layout
age_range_fig.update_layout(
    title=dict(text="Age Distribution of Customers", x=0.5, y=0.95, xanchor='center', yanchor='top'),
    xaxis=dict(title="Age"),
    yaxis=dict(title="Count"),
    showlegend=False,  # Hide legend
    bargap=0.1,        # Adjust gap between bars
    plot_bgcolor='white',  # Set background color
    #paper_bgcolor='lightgray',  # Set paper background color
    font=dict(family="Arial", size=12),  # Customize font
)

age_range_fig.show()


# In[30]:


age_bins = [0, 20, 30, 40, 50, 60, 100]
age_labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "61+"]

# Create a new column 'Age Group' based on age bins and labels
train['Age Group'] = pd.cut(train['Age'], bins=age_bins, labels=age_labels, right=False)


# In[31]:


age_group_counts = train['Age Group'].value_counts().reset_index()
age_group_counts.columns = ['Age Group', 'Count']
age_group_counts.style.background_gradient(cmap='Blues')


# In[32]:


fig = px.bar(
    age_group_counts,
    x='Age Group',
    y='Count',
    title='Age Group Analysis',
    labels={'Count': 'Number of Customers'},
    color='Age Group',
    color_discrete_sequence=px.colors.sequential.Blues[::-1],  # Customize the color
)

# Add number of customers on top of each bar
fig.update_traces(
    text=age_group_counts['Count'],  # Text to display on each bar
    textposition='outside',  # Position the text outside the bar
    marker=dict(line=dict(color='#000000', width=1)),  # Add black borders to bars
)
fig.update_layout(
    xaxis_title='Age Group',
    yaxis_title='Number of Customers',
    font=dict(size=12),
    title_font=dict(size=16),
    showlegend=False,
    #paper_bgcolor='#F2F2F2',  # Background color
    plot_bgcolor='#FFFFFF',  # Plot background color
    margin=dict(l=40, r=40, t=80, b=40),  # Margins
)

# Show the bar plot
fig.show()


# In[33]:


total_customers = len(train)
age_group_counts['Percentage'] = (age_group_counts['Count'] / total_customers) * 100

# Create a bar plot for age group analysis with percentages
fig = px.bar(
    age_group_counts,
    x='Age Group',
    y='Percentage',
    title='Age Group Analysis (Percentage of Total Customers)',
    labels={'Percentage': 'Percentage of Total Customers'},
    color='Age Group',
    color_discrete_sequence=px.colors.sequential.Blues[::-1],  # Customize the color
)
fig.update_layout(
    xaxis_title='Age Group',
    yaxis_title='Percentage of Total Customers',
    font=dict(size=12),
    title_font=dict(size=16),
    showlegend=False,
    #paper_bgcolor='#F2F2F2',  # Background color
    plot_bgcolor='#FFFFFF',  # Plot background color
    margin=dict(l=40, r=40, t=80, b=40),  # Margins
)

# Add percentage labels on top of each bar
for i in range(len(age_group_counts)):
    fig.add_annotation(
        x=age_group_counts['Age Group'][i],
        y=age_group_counts['Percentage'][i],
        text=f'{age_group_counts["Percentage"][i]:.2f}%',  # Format as a percentage
        #showarrow=False,
        font=dict(size=11),
        yshift=10,
    )
fig.update_traces(
    marker=dict(line=dict(color='#000000', width=1)),  # Add black borders to bars
)

# Show the bar plot
fig.show()    


# In[34]:


Age_Groups = train.groupby('Age Group')[['Unnamed: 0']].count()
fig = px.pie(Age_Groups,
            values = 'Unnamed: 0',
            names=Age_Groups.index,
            width=800,
            height=500,
            color_discrete_sequence = px.colors.sequential.Blues[6:4:-1])
fig.update_layout(
    title='Distribution of age Groups',
    legend_title='Age group',
    font=dict(size=14)
)
fig.layout.template = 'plotly'
fig.show()


# In[35]:


calculate_percentage_cross_tab_with_style(train,'Age Group')


# In[36]:


create_custom_bar_plot(train,'Age Group','satisfaction','#08306B','#98F5FF')


# In[37]:


satisfaction_by_gender_age = train.groupby(['Gender','Age Group'])['satisfaction'].value_counts(normalize=True).unstack()
satisfaction_by_gender_age.style.background_gradient(cmap='Blues')


# In[38]:


age_gender_satisfaction_percentages = (
    train.groupby(['Age Group', 'Gender', 'satisfaction'])['satisfaction']
    .count()
    .groupby(level=[0, 1])
    .apply(lambda x: 100 * x / x.sum())
    .reset_index(name='Percentage')
)

age_gender_satisfaction_percentages = age_gender_satisfaction_percentages.sort_values(
    by=['Age Group', 'Gender', 'satisfaction'], ascending=[True, True, False]
)

# Create a grouped bar plot to visualize the relationship between age, gender, and satisfaction percentages
fig = px.bar(
    age_gender_satisfaction_percentages,
    x='Age Group',
    y='Percentage',
    color='satisfaction',  # Color bars based on gender
    barmode='group',
    facet_col='Gender',  # Facet by satisfaction level
    title='Relationship Between Age, Gender, and Satisfaction',
    labels={'Percentage': 'Percentage of Customers'},
    color_discrete_sequence=['#98F5FF', '#193EB0'],  # Custom colors for gender
)
# Update the layout for aesthetics
fig.update_xaxes(categoryorder='array', categoryarray=age_labels)  # Set the order of age groups

# Customize the plot appearance
fig.update_layout(
    plot_bgcolor='white',  # Background color
    paper_bgcolor='white',  # Plot area background color
)

# Show the plot
fig.show()


# In[39]:


# Calculate the counts of each customer type
customer_type_counts = train['Customer Type'].value_counts()

# Values and labels for the pie chart
values = customer_type_counts.values
labels = customer_type_counts.index

# Custom blue colors
colors = ['#BFEFFF', '#1E90FE']

# Create the pie chart
fig = go.Figure(data=go.Pie(values=values, labels=labels, pull=[0.01, 0.05, 0.01, 0.05], hole=0.45, marker_colors=colors))

# Update the hover and text info
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)

# Add annotation for the year
fig.add_annotation(x=0.5, y=0.5, text='Customer Type',
                   font=dict(size=18, family='Verdana', color='black'), showarrow=False)

# Update the layout
fig.update_layout(title_text='Customer Type Distribution', title_font=dict(size=25, family='Verdana'))

# Show the pie chart
fig.show()


# In[40]:


calculate_percentage_cross_tab_with_style(train, 'Customer Type')


# In[41]:


create_custom_bar_plot(train, 'Customer Type', 'satisfaction',  '#08306B', '#98F5FF')


# In[42]:


# Group the data by 'Customer Type' and calculate the average age
average_age_by_customer_type = train.groupby('Customer Type')['Age'].mean().reset_index()
# Apply a blue color gradient to the 'Age' column
average_age_by_customer_type.style.background_gradient(cmap='Blues')


# In[43]:


# Define a custom color palette
custom_colors = ['#98F5FF', '#08306B'] #2980b9

# Create a box plot using Plotly with custom colors and additional style
fig = px.box(
    train, x='Customer Type', y='Age', color='Customer Type',
    title='Age Distribution by Customer Type',
    labels={'Age': 'Age', 'Customer Type': 'Customer Type'},
    color_discrete_sequence=custom_colors
)

# Customize plot appearance
fig.update_traces(marker_line_color='black', marker_line_width=1.5)
fig.update_xaxes(title_text='', showgrid=True)
fig.update_yaxes(title_text='Age', showgrid=True, gridwidth=0.5)
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=12),
    xaxis=dict(tickfont=dict(size=11)),
    yaxis=dict(tickfont=dict(size=11)),
    legend=dict(title='', orientation='h', x=0.5, y=1.15, font=dict(size=11)),
    title_font=dict(size=16, family='Arial', color='black'),
)

fig.show()


# In[44]:


# Calculate the distribution of Type of Travel
class_distribution = train['Type of Travel'].value_counts().reset_index()
class_distribution.columns = ['Type of Travel', 'Count']
class_distribution.style.background_gradient(cmap='Blues')


# In[45]:


# Calculate the distribution of travel classes
class_distribution = train['Class'].value_counts().reset_index()
class_distribution.columns = ['Class', 'Count']
class_distribution.style.background_gradient(cmap='Blues')


# In[46]:


type_of_travel_counts = train['Type of Travel'].value_counts()
travel_class_counts = train['Class'].value_counts()

# Create subplots with two columns
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]])

# Add pie chart for Type of Travel in the first column
fig.add_trace(go.Pie(
    labels=type_of_travel_counts.index,
    values=type_of_travel_counts.values,
    hole=0.4, pull=[0.01, 0.05, 0.01, 0.05],
    marker_colors=['#90EE90', '#08306B'],
    textinfo='percent+label',
    hoverinfo='label+percent',
), row=1, col=1)

# Add pie chart for Travel Class in the second column
fig.add_trace(go.Pie(
    labels=travel_class_counts.index,
    values=travel_class_counts.values,
    hole=0.4, pull=[0.02, 0.02, 0, 0],
    marker_colors=['#90EE90', '#66B2FF', '#08306B'],
    textinfo='percent+label',
    hoverinfo='label+percent',
), row=1, col=2)

# Add annotations to the pie charts
fig.add_annotation(
    text="Type of Travel",
    x=0.14,
    y=0.5,
    font=dict(size=19, color="black"),
    showarrow=False,
)

fig.add_annotation(
    text="Travel Class",
    x=0.85,
    y=0.5,
    font=dict(size=20, color="black"),
    showarrow=False,
)

# Update subplot titles and layout
fig.update_layout(
    title_text='Travel Information Distribution',
    title_font=dict(size=25, family='Verdana'),
    showlegend=False,
)

# Show the subplot
fig.show()


# In[47]:


calculate_percentage_cross_tab_with_style(train, 'Type of Travel')


# In[48]:


create_grouped_bar_chart('Type of Travel', 'satisfaction', train, '#08306B', '#90EE90')


# In[49]:


calculate_percentage_cross_tab_with_style(train, 'Class')


# In[50]:


create_grouped_bar_chart('Class', 'satisfaction', train, '#08306B', '#90EE90')


# In[51]:


# Create a histogram of flight distance
fig = px.histogram(
    train,
    x="Flight Distance",
    title="Distribution of Flight Distance",
    labels={"Flight Distance": "Distance (Miles)", "count": "Number of Customers"},
    color_discrete_sequence=["#08306B"],  # Custom color #87CEEB
)

# Customize the layout and appearance
fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
fig.update_yaxes(title_text="Number of Customers", title_font=dict(size=14), tickfont=dict(size=12))
fig.update_layout(
    title_font=dict(size=18),
    paper_bgcolor="rgba(255, 255, 255, 0.8)",  # Light background
    plot_bgcolor="rgba(255, 255, 255, 0.8)",   # Light plot area background
    font=dict(size=12, color="#333333"),  # Text color
    xaxis=dict(gridcolor="rgba(200, 200, 200, 0.2)"),  # Gridlines
    yaxis=dict(gridcolor="rgba(200, 200, 200, 0.2)"),  # Gridlines
)

# Show the plot
fig.show()


# In[52]:


# Create separate DataFrames for Satisfied and Dissatisfied customers
satisfied_data = train[train['satisfaction'] == 'satisfied']
dissatisfied_data = train[train['satisfaction'] == 'neutral or dissatisfied']

# Custom color palette
custom_palette = ['#90EE90', '#08306B']

# Create KDE plots for Satisfaction and Flight Distance Group
fig = make_subplots(rows=1, cols=2, subplot_titles=['Satisfied', 'Dissatisfied'])

# Add KDE plots for Satisfaction (Satisfied and Dissatisfied)
for data, col_idx, title in zip([satisfied_data, dissatisfied_data], [1, 2], ['Satisfied', 'Dissatisfied']):
    kde_trace = go.Histogram(x=data['Flight Distance'], nbinsx=20, histnorm='probability density', name='KDE',
                             marker_color=custom_palette[col_idx - 1])
    fig.add_trace(kde_trace, row=1, col=col_idx)

# Update layout for Satisfaction KDE plots
fig.update_layout(
    title_text='Kernel Density Estimation (KDE) of Flight Distance by Satisfaction',
    xaxis_title='Flight Distance',
    yaxis_title='Density',
    showlegend=False
)

# Show the plot for Satisfaction KDE plots
fig.show()


# In[53]:


train.groupby('Class')['Flight Distance'].mean().reset_index().style.background_gradient(cmap='Blues')


# In[54]:


# Define a custom color palette
custom_colors = ['#08306B', '#90EE90', '#66B2FF']  

# Create a box plot using Plotly Express
fig = px.box(train, x='Class', y='Flight Distance', color='Class',
             title='Flight Distance Distribution by Class of Travel',
             labels={'Flight Distance': 'Distance'},
             color_discrete_sequence=custom_colors
)

# Customize the plot appearance
fig.update_layout(
    xaxis_title='Class',
    yaxis_title='Distance',
    legend_title='',
)
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=12),
    xaxis=dict(tickfont=dict(size=11)),
    yaxis=dict(tickfont=dict(size=11)),
    legend=dict(title='', orientation='h', x=0.5, y=1.15, font=dict(size=11)),
    title_font=dict(size=16, family='Arial', color='black'),
)

# Show the box plot
fig.show()


# In[55]:


# Define bins and labels for flight distance groups
flight_distance_bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000]
flight_distance_labels = ["0-500", "501-1000", "1001-1500", "1501-2000", "2001-2500", "2501-3000", "3001-3500", "3501-4000", "4001-5000"]

# Create a new column 'Flight Distance Group' based on flight distance bins and labels
train['Flight Distance Group'] = pd.cut(train['Flight Distance'], bins=flight_distance_bins, labels=flight_distance_labels, right=False)


# In[56]:


# Group the data by flight distance group and count the number of customers in each group
flight_distance_group_counts = train['Flight Distance Group'].value_counts().reset_index()
flight_distance_group_counts.columns = ['Flight Distance Group', 'Count']
flight_distance_group_counts.style.background_gradient(cmap='Blues')


# In[57]:


# Calculate the percentage of customers in each flight distance group
total_customers = flight_distance_group_counts['Count'].sum()
flight_distance_group_counts['Percentage'] = (flight_distance_group_counts['Count'] / total_customers) * 100

# Create a bar plot for flight distance group analysis with percentages
fig = px.bar(
    flight_distance_group_counts,
    x='Flight Distance Group',
    y='Percentage',
    title='Flight Distance Group Analysis (Percentage)',
    labels={'Percentage': 'Percentage of Customers'},
    color='Flight Distance Group',
    color_discrete_sequence=px.colors.sequential.Blues[::-1]
)

# Add percentage values on top of each bar
fig.update_traces(
    text=flight_distance_group_counts['Percentage'].round(2).astype(str) + '%',  # Formatted percentage text
    textposition='outside',
    marker=dict(line=dict(color='#000000', width=1))
)

# Customize layout and style
fig.update_layout(
    xaxis_title='Flight Distance Group',
    yaxis_title='Percentage of Customers',
    font=dict(size=12),
    title_font=dict(size=16),
    showlegend=False,
    plot_bgcolor='#FFFFFF',
    margin=dict(l=40, r=40, t=80, b=40),
)

# Show the bar plot with percentages
fig.show()


# In[58]:


# Create a bar plot for flight distance group analysis
fig = px.bar(
    flight_distance_group_counts,
    x='Flight Distance Group',
    y='Count',
    title='Flight Distance Group Analysis',
    labels={'Count': 'Number of Customers'},
    #color_discrete_sequence=["#66B2FF"],  # Customize the color
    color='Flight Distance Group',
    color_discrete_sequence=px.colors.sequential.Blues[::-1] # Customize the color
)

# Add number of customers on top of each bar
fig.update_traces(
    text=flight_distance_group_counts['Count'],  # Text to display on each bar
    textposition='outside',  # Position the text outside the bar
    marker=dict(line=dict(color='#000000', width=1)),  # Add black borders to bars
)

# Customize layout and style
fig.update_layout(
    xaxis_title='Flight Distance Group',
    yaxis_title='Number of Customers',
    font=dict(size=12),
    title_font=dict(size=16),
    showlegend=False,
    #paper_bgcolor='#F2F2F2',  # Background color
    plot_bgcolor='#FFFFFF',  # Plot background color
    margin=dict(l=40, r=40, t=80, b=40),  # Margins
)

# Show the bar plot
fig.show()


# In[59]:


calculate_percentage_cross_tab_with_style(train, 'Flight Distance Group')


# In[60]:


create_custom_bar_plot(train, 'Flight Distance Group', 'satisfaction',  '#08306B', 'lightgreen')


# In[61]:


train.columns


# In[62]:


Services=['Inflight wifi service','Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness']


# In[63]:


# Define colors for 'satisfied' and 'neutral or dissatisfied'
dissatisfied_color = '#90EE90'
satisfied_color = '#66B2FF'#95B7D5

for i in range(len(Services)):
    # Call the create_grouped_bar_chart function with the current index
    create_grouped_bar_chart(Services[i], 'satisfaction', train, satisfied_color, dissatisfied_color)


# In[64]:


# Define a custom color palette
custom_palette = ['#90EE90', '#66B2FF']

# Create a scatter plot using Plotly Express with the custom color palette
fig = px.scatter(train, x='Departure Delay in Minutes', y='Arrival Delay in Minutes', color='satisfaction',
                 title='Scatter Plot of Departure Delay vs. Arrival Delay (Colored by Satisfaction)',
                 labels={'Departure Delay in Minutes': 'Departure Delay', 'Arrival Delay in Minutes': 'Arrival Delay'},
                 color_discrete_sequence=custom_palette)  # Use the custom color palette

# Show the Plotly figure
fig.show()


# In[65]:


# Group the data by both age group and flight distance group and count the number of customers in each group
grouped_counts = train.groupby(['Age Group', 'Flight Distance Group']).size().reset_index(name='Count')

# Create a grouped bar plot
fig = px.bar(
    grouped_counts,
    x='Age Group',
    y='Count',
    color='Flight Distance Group',
    barmode='group',
    title='Age Group vs. Flight Distance Group Analysis',
    labels={'Count': 'Number of Customers'},
    category_orders={"Age Group": age_labels},  # Order age groups
    color_discrete_sequence=px.colors.sequential.Blues[2::],  # Set the color palette
)

# Customize layout and style
fig.update_xaxes(title_text='Age Group', tickfont=dict(size=12))
fig.update_yaxes(title_text='Number of Customers', tickfont=dict(size=12))
fig.update_layout(
    title_font=dict(size=16),
    legend_title=dict(text='Flight Distance Group'),
    legend_font=dict(size=12),
    #paper_bgcolor='#F2F2F2',  # Background color
    plot_bgcolor='#FFFFFF',  # Plot background color
    margin=dict(l=40, r=40, t=80, b=40),  # Margins
)

# Show the grouped bar plot
fig.show()


# In[66]:


# Define age group bins and labels
age_bins = [0, 30, 45, 60, 100]
age_labels = ["Younger", "Middle", "Older", "Senior"]

# Create a new column 'Age Category' based on age bins and labels
train['Age Category'] = pd.cut(train['Age'], bins=age_bins, labels=age_labels, right=False)


# In[67]:


# Group data by different variables and calculate average satisfaction rates
group_vars = ['Age Category', 'Flight Distance Group']
satisfaction_by_group = train.groupby(group_vars)['satisfaction'].value_counts(normalize=True).unstack().reset_index()
satisfaction_by_group['neutral or dissatisfied'] = satisfaction_by_group['neutral or dissatisfied'] * 100
satisfaction_by_group['satisfied'] = satisfaction_by_group['satisfied'] * 100

# Convert to long format for Plotly
satisfaction_by_group_long = pd.melt(satisfaction_by_group, id_vars=group_vars, value_vars=['neutral or dissatisfied', 'satisfied'], 
                                   var_name='Satisfaction', value_name='SatisfactionRate')

# Define the desired color sequence
color_discrete_sequence = ['#66B2FF', '#90EE90'] 

# Create a bar plot using Plotly
fig = px.bar(satisfaction_by_group_long, x=group_vars[0], y='SatisfactionRate', color='Satisfaction', 
             facet_col=group_vars[1], facet_col_wrap=2,
             labels={group_vars[0]: group_vars[0], 'SatisfactionRate': 'Satisfaction Rate (%)'},
             title=f'Satisfaction Rate Based on {group_vars[0]} and {group_vars[1]}',
             color_discrete_sequence=color_discrete_sequence)

# Customize the layout
fig.update_xaxes(categoryorder='total ascending')
fig.update_layout(xaxis_title='Age Category', yaxis_title='Satisfaction Rate (%)',height=3000)
fig.update_yaxes(matches='y')
fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')

# Show the Plotly figure
fig.show()


# In[68]:


# Group data by Age Category, Flight Distance Group, and Class, and calculate counts
age_flight_class_counts = train.groupby(['Age Category', 'Flight Distance Group', 'Class'])['Unnamed: 0'].count().reset_index()

# Create a reversed "Blues" color palette
reversed_blues_palette = px.colors.sequential.Blues[::-1]

# Create a sunburst chart using Plotly Express with the reversed "Blues" color palette
fig = px.sunburst(age_flight_class_counts, path=['Age Category', 'Flight Distance Group', 'Class'], values='Unnamed: 0',
                  title='Sunburst Chart of Age, Flight Distance, and Class',
                  color_discrete_sequence=reversed_blues_palette)

# Show the Plotly figure
fig.show()


# In[69]:


# Calculate the average age for Personal and Business travelers
average_age_by_purpose = train.groupby('Type of Travel')['Age'].mean().reset_index()
average_age_by_purpose.style.background_gradient(cmap='Blues')


# In[70]:


# Create a bar plot to compare average age by travel purpose
fig = px.bar(average_age_by_purpose, x='Type of Travel', y='Age',
             title='Average Age by Travel Purpose',
             labels={'Age': 'Average Age', 'Type of Travel': 'Travel Purpose'},
             color='Type of Travel',  # Color bars by travel purpose
             color_discrete_sequence=['#66B2FF', '#90EE90'],  # Custom colors for travel purposes
             )

# Update the layout
fig.update_xaxes(categoryorder='array', title_text='Travel Purpose')  # Set the order and title of travel purposes

# Show the bar plot
fig.show()


# In[71]:


# Create a box plot to compare flight distance by travel purpose
fig = px.box(train, x='Type of Travel', y='Flight Distance',
             title='Flight Distance by Travel Purpose',
             labels={'Flight Distance': 'Flight Distance', 'Type of Travel': 'Travel Purpose'},
             color='Type of Travel',  # Color boxes by travel purpose
             color_discrete_sequence=['#66B2FF', '#90EE90'],  # Custom colors for travel purposes
             category_orders={'Type of Travel': ['Business', 'Personal Travel']},  # Order travel purposes
             )

# Update the layout
fig.update_xaxes(categoryorder='array', title_text='Travel Purpose')  # Set the order and title of travel purposes

# Show the box plot
fig.show()


# In[72]:


# Create a scatter plot to compare departure delay and arrival delay by travel purpose
fig = px.scatter(train, x='Departure Delay in Minutes', y='Arrival Delay in Minutes',
                 title='Departure Delay vs. Arrival Delay by Travel Purpose',
                 labels={'Departure Delay in Minutes': 'Departure Delay', 'Arrival Delay in Minutes': 'Arrival Delay',
                         'Type of Travel': 'Travel Purpose'},
                 color='Type of Travel',  # Color points by travel purpose
                 color_discrete_sequence=['#08306B', '#90EE90'],  # Custom colors for travel purposes
                 )

# Update the layout
fig.update_xaxes(title_text='Departure Delay')
fig.update_yaxes(title_text='Arrival Delay')
fig.update_xaxes(title_text='Travel Purpose', categoryorder='array', title_standoff=20)  # Set the order and title of travel purposes

# Show the scatter plot
fig.show()


# In[73]:


# Create histograms for age distribution for male and female customers
fig = px.histogram(train, x='Age', color='Gender', marginal='box', 
                   title='Age Distribution by Gender',
                   labels={'Age': 'Age'},
                   color_discrete_sequence=['#66B2FF', '#FF69B4'])  # Custom colors for gender

# Customize the plot appearance
fig.update_layout(
    xaxis_title='Age',
    yaxis_title='Count',
    legend_title='Gender',
    bargap=0.1, # Adjust the gap between bars
    plot_bgcolor='white',  # Background color
    paper_bgcolor='white',  # Plot area background color
)

fig.show()


# In[74]:


# Create separate dataframes for male and female customers
male_customers = train[train['Gender'] == 'Male']
female_customers = train[train['Gender'] == 'Female']


# In[75]:


from scipy.stats import ttest_ind

# Example: Compare 'Inflight wifi service' between male and female customers
feature = 'Inflight wifi service'
t_stat, p_value = ttest_ind(male_customers[feature], female_customers[feature])
print(f'Test statistic: {t_stat}')
print(f'P-value: {p_value}')

if p_value < 0.05:
    print(f"There is a significant difference in {feature} between male and female customers.")
else:
    print(f"No significant difference in {feature} between male and female customers.")


# In[76]:


# Example: Create a box plot to compare 'Seat comfort' satisfaction ratings
fig = px.box(train, x='Gender', y='Seat comfort', color='Gender',
             title='Comparison of Seat Comfort Satisfaction between Male and Female Customers',
             labels={'Seat comfort': 'Seat Comfort Satisfaction'},
             color_discrete_sequence=['#66B2FF', '#FF69B4'])  # Custom colors for gender

fig.update_layout(
    plot_bgcolor='white',  # Background color
    paper_bgcolor='white',  # Plot area background color
)

# Show the plot
fig.show()


# In[ ]:




