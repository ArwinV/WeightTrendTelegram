# WeightTrend Bot

## Overview
WeightTrend Bot is a Telegram bot designed to help users track their weight trends and progress over time. It integrates with Garmin Connect to log weight entries and provides visualizations of weight trends, rolling averages, and noise-to-trend ratios. The bot also allows users to set weight goals and trend limits for better tracking.

## Features
- **Log Weight**: Automatically log weight entries to Garmin Connect.
- **Visualize Trends**: Generate plots for weight trends, rolling averages, and noise-to-trend ratios.
- **Set Goals**: Define weight goals and track progress toward them.
- **Trend Limits**: Set limits for acceptable weight change percentages (e.g., cutting or bulking).
- **Interactive Telegram Interface**: Use inline buttons to navigate between trend limits, trend clarity, and overview sections.

## Installation

### Prerequisites
- Python 3.11 or higher
- A Garmin Connect account
- A Telegram bot token (create one via BotFather)

### Steps
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/WeightTrend-py3p11.git
    cd WeightTrend-py3p11
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Create a `config.json` file** to store sensitive information:
    ```json
    {
      "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
      "garmin_username": "YOUR_GARMIN_USERNAME",
      "garmin_password": "YOUR_GARMIN_PASSWORD"
    }
    ```

5. **Run the bot**:
    ```bash
    python bot.py
    ```

## Usage

### Commands
- `/getupdate`: Generate and display the weight trend overview.
- `/setgoal [value]`: Set a weight goal (e.g., `/setgoal 70.5`).
- `/removegoal`: Remove the current weight goal.
- `/trendlimits [lower_limit] [upper_limit]`: Set trend limits for weight change percentages (e.g., `/trendlimits -0.5 0.5`).

### Logging Weight
Send a numeric message (e.g., `70.5`) to log your weight to Garmin Connect and generate an immediate update.

## File Structure
- `bot.py`: Main script to run the bot.
- `config.json`: Configuration file for sensitive information.
- `requirements.txt`: List of dependencies.
- Other files for plotting and Telegram bot logic.

## Example Output

### Telegram Bot Interface
#### Overview:
Inline Buttons:
- 📊 Trend Limits
- 🔍 Trend Clarity

#### After Clicking "Trend Limits":
Displays the current trend limits and allows adjustments.

#### After Clicking "Back to Overview":
Returns to the main overview section.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the bot.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Disclaimer
This bot is not affiliated with Garmin or Telegram. Use at your own risk.

## Final Note
Replace placeholders like `YOUR_TELEGRAM_BOT_TOKEN` and `YOUR_GARMIN_USERNAME` with your actual credentials before running the bot.