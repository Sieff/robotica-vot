# Vision Challenge

## Activate venv!

    source .venv/bin/activate

## Developing a tracker

First just use the test method to run the current working directory with a sequence

    vot test --sequence sequences/ball1 VCTracker

Then copy working directory to a new folder tracker_x and add the tracker to the ini file

## Evaluate

vot evaluate TrackerX TrackerY

## Render Analysis

vot analysis TrackerX TrackerY