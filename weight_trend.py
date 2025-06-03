import asyncio
import traceback
import garminconnect
import json
import argparse
from getpass import getpass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, CallbackQueryHandler, filters
from telegram.helpers import escape_markdown
import tracemalloc
import pandas as pd  # Import pandas for rolling averages
import os  # Import os for file operations

tracemalloc.start()

# Load configuration from config.json
def load_config():
    try:
        with open("config.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: config.json file not found.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: Failed to parse config.json.")
        exit(1)

# Load configuration
config = load_config()

# Replace sensitive information with values from config
TELEGRAM_TOKEN = config["telegram_token"]
GARMIN_USERNAME = config["garmin_username"]
GARMIN_PASSWORD = config["garmin_password"]
AUTHORIZED_USER_ID = config["authorized_user_id"]  # Load authorized user ID from config

GOAL_FILE = "goal.txt"
TREND_LIMITS_FILE = "trend_limits.txt"

def load_goal():
    """Load the goal weight from the goal file."""
    try:
        with open(GOAL_FILE, 'r') as file:
            goal = float(file.read().strip())
            return goal
    except (FileNotFoundError, ValueError):
        return None  # No goal set

def save_goal(goal):
    """Save the goal weight to the goal file."""
    with open(GOAL_FILE, 'w') as file:
        file.write(str(goal))

# Rename the existing remove_goal function
def delete_goal_file():
    """Remove the goal weight by deleting the goal file."""
    try:
        os.remove(GOAL_FILE)
    except FileNotFoundError:
        pass

def load_trend_limits():
    """Load the trend limits from the file."""
    try:
        with open(TREND_LIMITS_FILE, 'r') as file:
            limits = file.read().strip().split()
            return float(limits[0]), float(limits[1])
    except (FileNotFoundError, ValueError, IndexError):
        return None, None  # No limits set

def save_trend_limits(lower_limit, upper_limit):
    """Save the trend limits to the file."""
    with open(TREND_LIMITS_FILE, 'w') as file:
        file.write(f"{lower_limit} {upper_limit}")

def fetch_recent_weight_entries(use_file, file_path):
    """Fetch weight entries from file or Garmin API and return the data."""
    if use_file:
        # Load weight data from file
        try:
            with open(file_path, 'r') as file:
                weight_data = json.load(file)
                return weight_data
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON from file '{file_path}'.")
            return None

    try:
        # Initialize Garmin client
        client = garminconnect.Garmin(GARMIN_USERNAME, GARMIN_PASSWORD)
        client.login()

        # Calculate start date (6 months before today)
        start_date = (datetime.now() - timedelta(days=6*30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        # Fetch recent weight entries
        weight_data = client.get_body_composition(startdate=start_date, enddate=end_date)

        # Save weight data to file
        with open(file_path, 'w') as file:
            json.dump(weight_data, file)
            print(f"Weight data saved to '{file_path}'.")

        return weight_data

    except garminconnect.GarminConnectConnectionError:
        print("Error: Unable to connect to Garmin Connect.")
    except garminconnect.GarminConnectTooManyRequestsError:
        print("Error: Too many requests. Please try again later.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Calculate noise-to-trend ratio
def calculate_noise_to_trend(stddev, trend):
    """Calculate the noise-to-trend ratio."""
    return stddev / abs(trend) if trend != 0 else None
    
def process_weight_data(weight_data, output_file='weight_trend.png', lower_limit=None, upper_limit=None):
    """Process weight data, generate the plot, and save it to a file."""
    if not weight_data or 'dateWeightList' not in weight_data or not weight_data['dateWeightList']:
        return None, None, None

    # Extract dates and weights
    dates = [datetime.strptime(entry['calendarDate'], '%Y-%m-%d') for entry in weight_data['dateWeightList']]
    weights = [entry['weight'] / 1000 for entry in weight_data['dateWeightList']]  # Convert grams to kilograms

    # Sort dates and weights in ascending order
    sorted_data = sorted(zip(dates, weights))  # Sort by date
    dates, weights = zip(*sorted_data)  # Unzip into separate lists

    # Create a pandas DataFrame for rolling averages
    df = pd.DataFrame({'Date': dates, 'Weight': weights})
    df.set_index('Date', inplace=True)

    # Interpolate missing weight data
    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(full_date_range)
    df.index.name = 'Date'

    # Identify interpolated and measured data points
    df['IsInterpolated'] = df['Weight'].isna()  # Mark missing values before interpolation
    df['Weight'] = df['Weight'].interpolate(method='linear')  # Perform interpolation

    # Separate measured and interpolated data
    measured_data = df[df['IsInterpolated'] == False]
    interpolated_data = df[df['IsInterpolated'] == True]
    
    # Calculate rolling averages for 7-day, 14-day, and 30-day windows
    df['7-Day Avg'] = df['Weight'].rolling(window=7, min_periods=1).mean()
    df['14-Day Avg'] = df['Weight'].rolling(window=14, min_periods=1).mean()
    df['30-Day Avg'] = df['Weight'].rolling(window=30, min_periods=1).mean()

    # Calculate rolling standard deviation for 7-day, 14-day, and 30-day windows
    df['7-Day StdDev'] = df['Weight'].rolling(window=7, min_periods=1).std()
    df['14-Day StdDev'] = df['Weight'].rolling(window=14, min_periods=1).std()
    df['30-Day StdDev'] = df['Weight'].rolling(window=30, min_periods=1).std()

    # Calculate daily changes for the 7-day rolling average
    df['7-Day Change'] = df['7-Day Avg'].diff()

    # Print daily changes to the console
    #print("Daily Changes for 7-Day Rolling Average:")
    #for date, change in zip(df.index, df['7-Day Change']):
    #    print(f"{date.strftime('%Y-%m-%d')}: {change:.2f} kg/day" if not pd.isna(change) else f"{date.strftime('%Y-%m-%d')}: No change (insufficient data)")

    # Calculate rolling trends (weight change per week) using rolling averages
    def calculate_rolling_trend(df, column):
        rolling_trends = df[column].diff() * 7  # Convert daily change to weekly change
        percentage_trends = (rolling_trends / df[column]) * 100  # Convert to percentage change
        return df.index, rolling_trends, percentage_trends

    # Filter data for the last 3 months
    now = datetime.now()
    three_months_ago = now - timedelta(days=90)
    df_last_3_months = df[df.index >= three_months_ago]

    # Calculate rolling trends for the last 3 months
    trend_dates_7_3m, rolling_trends_7_3m, percentage_trends_7_3m = calculate_rolling_trend(df_last_3_months, '7-Day Avg')
    trend_dates_14_3m, rolling_trends_14_3m, percentage_trends_14_3m = calculate_rolling_trend(df_last_3_months, '14-Day Avg')
    trend_dates_30_3m, rolling_trends_30_3m, percentage_trends_30_3m = calculate_rolling_trend(df_last_3_months, '30-Day Avg')

    # Calculate rolling standard deviations and noise-to-trend ratios for the last 3 months
    stddev_7_3m = df_last_3_months['7-Day StdDev']
    stddev_14_3m = df_last_3_months['14-Day StdDev']
    stddev_30_3m = df_last_3_months['30-Day StdDev']

    # Ensure recent_7_day is defined before use
    recent_7_day = df_last_3_months['7-Day Avg'].diff().iloc[-1] * 7 if not df_last_3_months.empty else None
    recent_14_day = df_last_3_months['14-Day Avg'].diff().iloc[-1] * 7 if not df_last_3_months.empty else None
    recent_30_day = df_last_3_months['30-Day Avg'].diff().iloc[-1] * 7 if not df_last_3_months.empty else None
    noise_to_trend_7 = calculate_noise_to_trend(stddev_7_3m.iloc[-1], recent_7_day) if not stddev_7_3m.empty else None
    noise_to_trend_30 = calculate_noise_to_trend(stddev_30_3m.iloc[-1], recent_30_day) if not stddev_30_3m.empty else None
    noise_to_trend_30 = calculate_noise_to_trend(stddev_30_3m.iloc[-1], recent_30_day) if not stddev_30_3m.empty else None

    # Use the default MATLAB colors (tab10 colormap)
    colors = plt.cm.tab10.colors

    # Create subplots (vertical layout)
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 12))  # Adjusted for iPhone 15 Pro resolution

    # Plot the weight data (last 3 months)
    axs[0].plot(measured_data.index, measured_data['Weight'], marker='o', linestyle='-', color=colors[0], label='Measured Weight (kg)')  # MATLAB blue
    axs[0].plot(interpolated_data.index, interpolated_data['Weight'], marker='x', linestyle='None', color=colors[0], label='Interpolated Weight (kg)')  # MATLAB orange

    axs[0].set_xlim(three_months_ago, now)

    # Calculate the dates for 7 days, 14 days, and 30 days ago
    seven_days_ago = now - timedelta(days=7)
    fourteen_days_ago = now - timedelta(days=14)
    thirty_days_ago = now - timedelta(days=30)

    # Add vertical dashed lines for rolling trend ranges (no legend for these lines)
    axs[0].axvline(x=seven_days_ago, color=colors[2], linestyle='--', linewidth=1, alpha=0.8)  # MATLAB green
    axs[0].axvline(x=fourteen_days_ago, color=colors[3], linestyle='--', linewidth=1, alpha=0.8)  # MATLAB red
    axs[0].axvline(x=thirty_days_ago, color=colors[4], linestyle='--', linewidth=1, alpha=0.8)  # MATLAB purple

    # Set title, labels, and grid
    axs[0].set_title('Weight (Last 3 Months)', fontsize=14)
    axs[0].set_xlabel('Date', fontsize=12)
    axs[0].set_ylabel('Weight (kg)', fontsize=12)
    axs[0].grid(True, which='both', linestyle=':', linewidth=0.5)
    axs[0].legend(fontsize=10)

    # Format x-axis to exclude the year
    axs[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%b'))

    # Plot the rolling trends (last 3 months) with a second y-axis
    ax1 = axs[1]
    ax2 = ax1.twinx()  # Create a second y-axis

    # Plot rolling trends in kg/week
    line1, = ax1.plot(trend_dates_7_3m, rolling_trends_7_3m, linestyle='-', color=colors[2], label='7-Day Rolling Trend (kg/week)')  # MATLAB green
    line2, = ax1.plot(trend_dates_14_3m, rolling_trends_14_3m, linestyle='-', color=colors[3], label='14-Day Rolling Trend (kg/week)')  # MATLAB red
    line3, = ax1.plot(trend_dates_30_3m, rolling_trends_30_3m, linestyle='-', color=colors[4], label='30-Day Rolling Trend (kg/week)')  # MATLAB purple
    ax1.set_ylabel('Weight Change (kg/week)', fontsize=12)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)  # Dashed grid for the first y-axis

    # Plot percentage trends on the second y-axis
    line4, = ax2.plot(trend_dates_7_3m, percentage_trends_7_3m, linestyle='--', color=colors[2], alpha=0.5)  # MATLAB green
    line5, = ax2.plot(trend_dates_14_3m, percentage_trends_14_3m, linestyle='--', color=colors[3], alpha=0.5)  # MATLAB red
    line6, = ax2.plot(trend_dates_30_3m, percentage_trends_30_3m, linestyle='--', color=colors[4], alpha=0.5)  # MATLAB purple
    ax2.set_ylabel('Weight Change (%)', fontsize=12)

    # Add horizontal dashed lines for trend limits to the weight trend plot
    if lower_limit is not None and upper_limit is not None:
        ax2.axhline(y=lower_limit, color='black', linestyle='--', linewidth=1, alpha=0.8)  # Black horizontal line for lower limit
        ax2.axhline(y=upper_limit, color='black', linestyle='--', linewidth=1, alpha=0.8)  # Black horizontal line for upper limit

    # Add a dashed grid for the second y-axis with increased spacing
    ax2.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Dashed grid for the second y-axis
    for line in ax2.yaxis.get_gridlines():
        line.set_dashes((5, 10))  # Custom dash pattern: 5 pixels dash, 10 pixels space

    # Add legends for kg/week trends only
    lines_kg = [line1, line2, line3]  # Lines for kg/week trends
    labels_kg = [line.get_label() for line in lines_kg]  # Labels for kg/week trends
    ax1.legend(lines_kg, labels_kg, loc='upper left', fontsize=10)  # Only kg/week trends

    # Set title and x-axis formatting
    ax1.set_title('Rolling Trends (Last 3 Months)', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%b'))

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)  # High DPI for better resolution
    plt.close()

    # Calculate the most recent rolling trends
    recent_7_day = rolling_trends_7_3m.iloc[-1] if not rolling_trends_7_3m.empty else None
    recent_14_day = rolling_trends_14_3m.iloc[-1] if not rolling_trends_14_3m.empty else None
    recent_30_day = rolling_trends_30_3m.iloc[-1] if not rolling_trends_30_3m.empty else None

    # Calculate the most recent percentage trends
    recent_7_day_percentage = percentage_trends_7_3m.iloc[-1] if not percentage_trends_7_3m.empty else None
    recent_14_day_percentage = percentage_trends_14_3m.iloc[-1] if not percentage_trends_14_3m.empty else None
    recent_30_day_percentage = percentage_trends_30_3m.iloc[-1] if not percentage_trends_30_3m.empty else None

    return recent_7_day, recent_14_day, recent_30_day, recent_7_day_percentage, recent_14_day_percentage, recent_30_day_percentage, stddev_7_3m, stddev_14_3m, stddev_30_3m

def log_weight_to_garmin(weight_kg):
    """Log the given weight to Garmin Connect."""
    try:
        # Initialize Garmin client
        client = garminconnect.Garmin(GARMIN_USERNAME, GARMIN_PASSWORD)
        client.login()

        # Log weight to Garmin
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        client.add_weigh_in(weight=weight_kg)
        print(f"Weight {weight_kg:.2f} kg logged to Garmin successfully.")
        return True
    except garminconnect.GarminConnectConnectionError:
        print("Error: Unable to connect to Garmin Connect.")
    except garminconnect.GarminConnectTooManyRequestsError:
        print("Error: Too many requests. Please try again later.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return False

def escape_except_asterisk(text):
    """Escape all reserved MarkdownV2 characters except the asterisk."""
    reserved_characters = ['_', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in reserved_characters:
        text = text.replace(char, f"\\{char}")
    return text

# Telegram bot command handler
async def get_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /getupdate command."""
    user_id = update.message.from_user.id  # Get the user ID of the person sending the command

    # Check if the user is authorized
    if user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("Unauthorized access. You are not allowed to use this bot.")
        return

    # Fetch weight data
    weight_data = fetch_recent_weight_entries(use_file=False, file_path='weight_data.json')
    if not weight_data:
        await update.message.reply_text("No weight data available.")
        return

    # Process the weight data and generate the plot
    lower_limit, upper_limit = load_trend_limits()
    recent_7_day, recent_14_day, recent_30_day, recent_7_day_percentage, recent_14_day_percentage, recent_30_day_percentage, stddev_7_3m, stddev_14_3m, stddev_30_3m = process_weight_data(weight_data, lower_limit=lower_limit, upper_limit=upper_limit)

    # Get the current weight
    current_weight = weight_data['dateWeightList'][0]['weight'] / 1000  # Convert grams to kilograms

    # Load the goal weight
    goal_weight = load_goal()

    # Helper function to check if a trend is within limits
    def check_trend_within_limits(percentage):
        if lower_limit is not None and upper_limit is not None:
            if lower_limit <= percentage <= upper_limit: 
                return "✅"  # Green checkbox
            else:
                return "⚠️"  # Yellow triangle with exclamation mark
        return ""  # No limits set

    # Calculate weight loss to go and time to goal
    if goal_weight is not None:
        weight_loss_to_go = current_weight - goal_weight
        if recent_14_day < 0:  # Ensure weight is decreasing
            weeks_to_goal = abs(weight_loss_to_go / recent_14_day)
            days_to_goal = int(weeks_to_goal * 7)
            time_to_goal = f"{int(weeks_to_goal)} weeks and {days_to_goal % 7} days"
        else:
            time_to_goal = "Not achievable with the current trend"
    else:
        weight_loss_to_go = None
        time_to_goal = None

    # Calculate noise-to-trend ratios for the most recent trends
    noise_to_trend_7 = calculate_noise_to_trend(stddev_7_3m.iloc[-1], recent_7_day) if recent_7_day is not None else None
    noise_to_trend_14 = calculate_noise_to_trend(stddev_14_3m.iloc[-1], recent_14_day) if recent_14_day is not None else None
    noise_to_trend_30 = calculate_noise_to_trend(stddev_30_3m.iloc[-1], recent_30_day) if recent_30_day is not None else None

    # Send the plot
    chat_id = update.message.chat_id
    await context.bot.send_photo(chat_id=chat_id, photo=open('weight_trend.png', 'rb'))

    # Create the main message
    message = (
        f"*📊 Weight Progress Overview*\n\n"
        f"*Current Weight:* {current_weight:.2f} kg\n"
    )

    if goal_weight is not None:
        message += (
            f"*Goal Weight:* {goal_weight:.2f} kg\n"
            f"*Weight Loss to Go:* {weight_loss_to_go:.2f} kg\n"
            f"*Estimated Time to Goal:* {time_to_goal} (using 14-day trend)\n\n"
        )
    else:
        message += "*Goal Weight:* Not set. Use /setgoal [value] to set a goal.\n\n"

    # Add recent trends to the message
    message += (
        f"*📈 Recent Trends:*\n"
        f"• *7-Day:* {recent_7_day:.2f} kg/week ({recent_7_day_percentage:.2f}%) {check_trend_within_limits(recent_7_day_percentage)}\n"
        f"• *14-Day:* {recent_14_day:.2f} kg/week ({recent_14_day_percentage:.2f}%) {check_trend_within_limits(recent_14_day_percentage)}\n"
        f"• *30-Day:* {recent_30_day:.2f} kg/week ({recent_30_day_percentage:.2f}%) {check_trend_within_limits(recent_30_day_percentage)}\n\n"
    )

    # Create collapsible sections for trend limits and trend clarity
    trend_limits_message = (
        f"*📊 Current Trend Limits:*\n"
        f"• Lower Limit: {lower_limit:.2f}%\n"
        f"• Upper Limit: {upper_limit:.2f}%\n\n"
        if lower_limit is not None and upper_limit is not None
        else "*📊 Current Trend Limits:* Not set. Use /trendlimits [lower_limit] [upper_limit] to set limits.\n\n"
    )

    trend_clarity_message = (
        f"*🔍 Trend Clarity:*\n"
        f"• *7-Day:* Noise = {stddev_7_3m.iloc[-1]:.2f} kg, Noise/Trend = {noise_to_trend_7:.2f}\n"
        f"• *14-Day:* Noise = {stddev_14_3m.iloc[-1]:.2f} kg, Noise/Trend = {noise_to_trend_14:.2f}\n"
        f"• *30-Day:* Noise = {stddev_30_3m.iloc[-1]:.2f} kg, Noise/Trend = {noise_to_trend_30:.2f}\n\n"
    )

    # Pass messages to button_handler
    context.bot_data["trend_limits_message"] = trend_limits_message
    context.bot_data["trend_clarity_message"] = trend_clarity_message
    context.bot_data["overview_message"] = message

    # Create inline buttons for collapsible sections
    keyboard = [
        [InlineKeyboardButton("📊 Trend Limits", callback_data="trend_limits")],
        [InlineKeyboardButton("🔍 Trend Clarity", callback_data="trend_clarity")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send the main message with inline buttons
    await context.bot.send_message(chat_id=chat_id, text=escape_except_asterisk(message), parse_mode="MarkdownV2", reply_markup=reply_markup)

# Handle button clicks
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks for collapsible sections."""
    query = update.callback_query
    await query.answer()  # Acknowledge the button click

    # Retrieve messages from bot_data
    trend_limits_message = context.bot_data.get("trend_limits_message", "")
    trend_clarity_message = context.bot_data.get("trend_clarity_message", "")
    overview_message = context.bot_data.get("overview_message", "")

    # Check which button was pressed
    if query.data == "trend_limits":
        keyboard = [[InlineKeyboardButton("⬅️ Back to Overview", callback_data="overview")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text=escape_except_asterisk(trend_limits_message), parse_mode="MarkdownV2", reply_markup=reply_markup)
    elif query.data == "trend_clarity":
        keyboard = [[InlineKeyboardButton("⬅️ Back to Overview", callback_data="overview")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text=escape_except_asterisk(trend_clarity_message), parse_mode="MarkdownV2", reply_markup=reply_markup)
    elif query.data == "overview":
        keyboard = [
            [InlineKeyboardButton("📊 Trend Limits", callback_data="trend_limits")],
            [InlineKeyboardButton("🔍 Trend Clarity", callback_data="trend_clarity")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text=escape_except_asterisk(overview_message), parse_mode="MarkdownV2", reply_markup=reply_markup)

async def set_goal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /setgoal command."""
    user_id = update.message.from_user.id

    # Check if the user is authorized
    if user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("Unauthorized access. You are not allowed to use this bot.")
        return

    # Parse the goal value
    try:
        goal_weight = float(context.args[0])
        save_goal(goal_weight)
        await update.message.reply_text(f"Goal weight set to {goal_weight:.2f} kg.")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /setgoal [value] (e.g., /setgoal 70.5)")

async def remove_goal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /removegoal command."""
    await update.message.reply_text("Goal weight removed.")
    # Call the renamed function to delete the goal file
    delete_goal_file()
    await update.message.reply_text("Goal weight removed.")

async def set_trend_limits(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /trendlimits command."""
    user_id = update.message.from_user.id

    # Check if the user is authorized
    if user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("Unauthorized access. You are not allowed to use this bot.")
        return

    # Parse the lower and upper limits
    try:
        # Ensure that the lowest value is saved as the lower limit and the highest as the upper limit
        lower_limit = min(float(context.args[0]), float(context.args[1]))
        upper_limit = max(float(context.args[0]), float(context.args[1]))
        save_trend_limits(lower_limit, upper_limit)
        await update.message.reply_text(f"Trend limits set to {lower_limit:.2f}% and {upper_limit:.2f}%.")
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /trendlimits [lower_limit] [upper_limit] (e.g., /trendlimits -0.5 0.5)\nGood starting values:\n- Cutting: -0.5% to -1.0% / week.\n- Bulking: +0.25% to +0.5% / week\n")

async def log_weight(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle numeric messages to log weight to Garmin and send an immediate update."""
    user_id = update.message.from_user.id

    # Check if the user is authorized
    if user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("Unauthorized access. You are not allowed to use this bot.")
        return

    # Parse the weight from the message
    try:
        weight_kg = float(update.message.text)
        if weight_kg <= 0:
            raise ValueError("Weight must be a positive number.")

        # Log the weight to Garmin
        success = log_weight_to_garmin(weight_kg)
        if success:
            await update.message.reply_text(f"Weight {weight_kg:.2f} kg logged to Garmin successfully.\nCreating overview...")

            # Call the get_update function to send the update
            await get_update(update, context)
        else:
            await update.message.reply_text("Failed to log weight to Garmin. Please try again later.")
    except ValueError:
        await update.message.reply_text("Invalid input. Please send a valid weight in kilograms (e.g., 70.5).")

# Main function to start the bot
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("getupdate", get_update))
    application.add_handler(CommandHandler("setgoal", set_goal))
    application.add_handler(CommandHandler("removegoal", remove_goal))
    application.add_handler(CommandHandler("trendlimits", set_trend_limits))

    # Add message handler for logging weight
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, log_weight))

    # Add the button handler to the bot
    application.add_handler(CallbackQueryHandler(button_handler))

    # Start the bot
    application.run_polling()

if __name__ == "__main__":
    main()