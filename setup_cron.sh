#!/bin/bash

# Directory where main.py is located
SCRIPT_DIR="/home/yewen/BCM/TBRD/TBRD-null-pipeline/"

# Adding a cron job
(crontab -l 2>/dev/null; echo "0 3 * * 1 python $SCRIPT_DIR/main.py") | crontab -

echo "Cron job set to run main.py every Monday at 3:00 AM."
