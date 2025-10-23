#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRT Story Presentation Experiment Script

This script presents emotional prosody stories to participants and collects responses
to comprehension questions. Stories are played with audio, followed by question audio
and visual response options.

@author: Tong
"""
# %% Set up sound device
import os
os.environ['SD_ENABLE_ASIO'] = '1'
import sounddevice as sd
sd.query_hostapis()
sd.query_devices()
# %% Import libraries
import os
import numpy as np
import pandas as pd
from expyfun import ExperimentController, decimals_to_binary
from expyfun.io import read_wav
import threading

# %% Set parameters
fs = 48000  # Sample rate
n_channel = 2  # Stereo output
stim_db = 65  # Stimulus volume in dB

pause_dur = 1.0  # Pause duration between story and questions

# %% Load story and question data
# Set up paths - normalize to forward slashes for cross-platform compatibility
pilot_path = "C:/Users/Admin/Desktop/PRT_EEG_pilot/"
stim_path = pilot_path+"stim_normalized/"
csv_path = pilot_path+"code/stim_presentation/story_questions_mapping.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

# Load the CSV
df = pd.read_csv(csv_path)

# Get unique stories
stories_df = df.groupby('story_id').first().reset_index()
print(f"Found {len(stories_df)} stories to present")

# Display available stories
print("\nAvailable stories:")
for idx, story_row in stories_df.iterrows():
    print(f"  {idx + 1}. {story_row['story_name']} ({story_row['story_id']}) - {story_row['emotion']}")

# Ask user which story to start from
while True:
    start_input = input(f"\nEnter story number to start from (1-{len(stories_df)}, or press Enter for story 1): ").strip()

    if start_input == "":
        start_story_idx = 0
        break

    try:
        start_story_num = int(start_input)
        if 1 <= start_story_num <= len(stories_df):
            start_story_idx = start_story_num - 1
            break
        else:
            print(f"Please enter a number between 1 and {len(stories_df)}")
    except ValueError:
        print("Please enter a valid number")

print(f"\nStarting from story {start_story_idx + 1}: {stories_df.iloc[start_story_idx]['story_name']}")

# Ask if user wants to run sound check test trial (only if starting from story 1)
run_test_trial = False
if start_story_idx == 0:
    while True:
        test_input = input("\nDo you want to run the sound check test trial? (y/n): ").strip().lower()
        if test_input in ['y', 'yes']:
            run_test_trial = True
            print("Sound check test trial will be included.")
            break
        elif test_input in ['n', 'no']:
            run_test_trial = False
            print("Skipping sound check test trial.")
            break
        else:
            print("Please enter 'y' or 'n'")

# Filter stories from starting point
stories_df = stories_df.iloc[start_story_idx:].reset_index(drop=True)

# %% Preload all audio files
print("\nPreloading audio files...")

story_audio = {}
question_audio = {}

for idx, story_row in stories_df.iterrows():
    story_id = story_row['story_id']
    story_name = story_row['story_name']

    # Load story audio
    # Normalize to forward slashes (works on all platforms)
    story_file = f"{pilot_path}/{story_row['story_path']}"
    print(f"Loading story: {story_name} ({story_id})")
    try:
        temp, _ = read_wav(story_file)
        story_audio[story_id] = temp
        # Duration is number of samples (columns) divided by sample rate
        print(f"  ✓ Story loaded: {temp.shape[1]/fs:.1f}s")
    except Exception as e:
        print(f"  ✗ Error loading story: {e}")
        story_audio[story_id] = None

    # Load question audio for this story
    story_questions = df[df['story_id'] == story_id]
    question_audio[story_id] = []

    for _, q_row in story_questions.iterrows():
        # Normalize to forward slashes (works on all platforms)
        q_file = f"{pilot_path}/{q_row['question_audio_path']}"
        try:
            temp, _ = read_wav(q_file)
            question_audio[story_id].append({
                'audio': temp,
                'question_num': q_row['question_num'],
                'question_text': q_row['question_text'],
                'answer_options': q_row['answer_options']
            })
        except Exception as e:
            print(f"  ✗ Error loading question {q_row['question_num']}: {e}")
            question_audio[story_id].append({
                'audio': None,
                'question_num': q_row['question_num'],
                'question_text': q_row['question_text'],
                'answer_options': q_row['answer_options']
            })

print("Audio preloading completed!")

# %% Helper functions

def parse_answer_options(options_str):
    """Parse answer options from CSV format"""
    if options_str == "Free response":
        return None
    # Split by " | " and extract letter and text
    options = []
    for opt in options_str.split(" | "):
        options.append(opt.strip())
    return options

def show_loading_screen(ec, message="Loading...", submessage="Please wait..."):
    """Show a loading screen"""
    ec.screen_text(message, pos=[0, 0.2], units='norm', color='w')
    ec.screen_text(submessage, pos=[0, -0.2], units='norm', color='w')
    ec.flip()

def background_load_buffer(ec, audio_data, trial_id, callback=None):
    """Load audio buffer in background thread"""
    def load_thread():
        try:
            print(f"Background loading buffer for {trial_id}...")
            ec.load_buffer(audio_data)
            print(f"✓ Background loading completed for {trial_id}")
            if callback:
                callback(True, None)
        except Exception as e:
            print(f"✗ Background loading failed for {trial_id}: {e}")
            if callback:
                callback(False, str(e))

    thread = threading.Thread(target=load_thread)
    thread.daemon = True
    thread.start()
    return thread


# %% Experiment instructions
instruction_text_1 = """Welcome to the Listening Activity!\n\n

You will listen to 5 different stories.\n

After each story, you will answer 5 questions about what you heard.\n

"""

# instruction_text_2 = """
# For all questions, tell us your answer out loud.

# Some questions will have choices, and some will not. 

# Say your answer out loud, and speak loudly and clearly.

# """

instruction_text_3 = """

Just relax and listen carefully to each story!\n

Take your time and do your best.\n

"""

first_story_instruction = """Great! We are ready to start.\n

Remember:

- Listen carefully to each story\n

- When the story plays, keep your eyes on the cross (+) in the middle of the screen.\n

- Do your best to stay as still as you can while the stories are playing.\n

Let's begin with the first story!\n

"""

# %% Experiment setup
ec_args = dict(
    exp_name='PRT_Stories',
    window_size=[2560, 1440],
    session='00',
    full_screen=True,
    n_channels=n_channel,
    version='dev',
    stim_fs=fs,
    stim_db=stim_db,
    force_quit=['end']
)

# %% Run experiment
n_bits_story = int(np.ceil(np.log2(5))) # 5 stories
n_bits_question = int(np.ceil(np.log2(5)))  # Max 5 questions per story

with ExperimentController(**ec_args) as ec:
    # Show initial instructions (3 screens)
    ec.screen_prompt(instruction_text_1, live_keys=['space'])
    # ec.screen_prompt(instruction_text_2, live_keys=['space'])
    ec.screen_prompt(instruction_text_3, live_keys=['space'])

    # Test trial - play first 10 seconds of first story for sound check (if requested)
    if run_test_trial:
        # Preload first story for sound check
        first_story_id = stories_df.iloc[0]['story_id']
        print(f"\nPreloading first story for sound check: {first_story_id}")
        show_loading_screen(ec, "Preparing sound check...")
        if story_audio[first_story_id] is not None:
            ec.load_buffer(story_audio[first_story_id])

        ec.screen_text("Sound Check", pos=[0, 0.3], units='norm', color='w', font_size=32)
        ec.screen_text("We will play 10 seconds of the first story to check the sound.",
                      pos=[0, 0], units='norm', color='w', font_size=24, wrap=True)
        ec.flip()
        ec.wait_one_press(max_wait=np.inf, live_keys=['space'])

        # Show fixation cross
        ec.screen_text("+", pos=(0.75, 0), units='norm', color='white', font_size=64)
        ec.flip()
        ec.wait_secs(1.0)

        # Identify trial before starting
        ec.identify_trial(ec_id="sound_check", ttl_id=[])

        # Play first 10 seconds
        print("\nPlaying 10-second sound check...")
        test_duration = 10.0  # 10 seconds
        test_start_time = ec.start_stimulus()

        # Keep cross visible during playback
        ec.screen_text("+", pos=(0.75, 0), units='norm', color='white', font_size=64)
        ec.flip()

        # Wait for 10 seconds
        while ec.current_time < test_start_time + test_duration:
            ec.check_force_quit()
            ec.wait_secs(0.1)

        ec.stop()
        ec.trial_ok()
        print("Sound check completed!")

        # Ask if sound is okay
        ec.screen_prompt("You should have heard the sounds coming from both (R/L) headphones.", live_keys=['space'])

    # Show instruction before first story
    ec.screen_prompt(first_story_instruction, live_keys=['space'])

    trial_start_time = -np.inf

    # Loop through each story
    for story_idx, story_row in stories_df.iterrows():
        story_id = story_row['story_id']
        story_name = story_row['story_name']
        emotion = story_row['emotion']

        print(f"\n{'='*60}")
        print(f"Story {story_idx + 1}/{len(stories_df)}: {story_name} ({emotion})")
        print(f"{'='*60}")

        # Display story prompt
        ec.screen_text(f"Story {story_idx + 1} of {len(stories_df)}: {story_name}",
                      pos=[0, 0.2], units='norm', color='w')
        ec.flip()

        # Wait for space press to start story
        ec.wait_one_press(max_wait=np.inf, live_keys=['space'])

        # Play story
        if story_audio[story_id] is not None:
            # Load story buffer (first story already loaded in preload section)
            show_loading_screen(ec, f"Loading story {story_idx + 1}...")
            ec.load_buffer(story_audio[story_id])

            # Duration is number of samples (shape[1]) divided by sample rate
            story_duration = story_audio[story_id].shape[1] / fs

            # Show fixation cross for 2 seconds before story starts
            ec.screen_text("+", pos=(0.75, 0), units='norm', color='white', font_size=64)
            ec.flip()
            ec.wait_secs(1.0)

            # Identify trial
            trial_id = f"{story_id}_story"
            ec.identify_trial(ec_id=trial_id, ttl_id=[])

            # Start playback immediately (no wait needed between stories)
            trial_start_time = ec.start_stimulus()

            # Redraw cross after starting stimulus so it stays on screen during story
            ec.screen_text("+", pos=(0.75, 0), units='norm', color='white', font_size=64)
            ec.flip()
            ec.wait_secs(0.1)

            # Trigger
            ec.stamp_triggers([(b + 1) * 4 for b in decimals_to_binary([story_idx], [n_bits_story])])

            # Wait for story to finish - cross stays on screen the whole time
            while ec.current_time < trial_start_time + story_duration:
            #to skip story playing uncomment the line below and comment the one above
            #while ec.current_time < trial_start_time + 0.5:
                ec.check_force_quit()
                ec.wait_secs(0.1)

            ec.stop()
            ec.trial_ok()

            # Write story data
            ec.write_data_line("story", {
                "story_num": story_idx,
                "story_id": story_id,
                "story_name": story_name,
                "emotion": emotion,
                "duration": story_duration
            })

        # Pause before questions
        ec.wait_secs(pause_dur)

        # Show transition to questions (only before first question)
        ec.screen_text(f"Story finished!", pos=[0, 0.2], units='norm', color='w')
        ec.screen_text(f" Now you will answer some questions about this story.\n You will see the question on the screen and you will also hear the question.\n Please wait until the whole question is read aloud and then tell us your answer.",
                      pos=[0, -0.1], units='norm', color='w')

        ec.flip()
        ec.wait_one_press(max_wait=np.inf, live_keys=['space'])

        # Present questions for this story
        questions = question_audio[story_id]
        story_responses = {}

        for q_idx, q_data in enumerate(questions):
            print(f"\n  Question {q_data['question_num']}: {q_data['question_text'][:50]}...")

            # Parse answer options first to determine display layout
            options = parse_answer_options(q_data['answer_options'])

            if q_data['audio'] is not None:
                # Load audio
                ec.load_buffer(q_data['audio'])
                question_duration = q_data['audio'].shape[1] / fs

                # Identify trial
                q_trial_id = f"{story_id}_q{q_data['question_num']}_play"
                ec.identify_trial(ec_id=q_trial_id, ttl_id=[])

                # Start audio playback immediately
                trial_start_time = ec.start_stimulus()

                # Display question and options RIGHT AFTER starting audio
                ec.screen_text(f"Question {q_data['question_num']} of 5",
                              pos=[0, 0.8], units='norm', color='w', font_size=18)

                if options is None:
                    # Free response
                    ec.screen_text(q_data['question_text'], pos=[0, 0.4], units='norm',
                                  color='w', font_size=32, wrap=True)
                    # ec.screen_text("Answer out loud after the question finishes.",
                    #               pos=[0, -0.1], units='norm', color='yellow', font_size=22)
                else:
                    # Multiple choice
                    ec.screen_text(q_data['question_text'], pos=[0, 0.5], units='norm',
                                  color='w', font_size=28, wrap=True)
                    # ec.screen_text("Read the answer choices after the question finishes:",
                    #               pos=[0, 0.25], units='norm', color='yellow', font_size=22)

                    # Display answer options
                    y_start = 0.0
                    y_spacing = 0.15
                    for i, option_text in enumerate(options):
                        y_pos = y_start - (i * y_spacing)
                        ec.screen_text(option_text, pos=[0, y_pos], units='norm',
                                      color='w', font_size=24, wrap=True)

                ec.flip()
                ec.wait_secs(0.1)

                # Trigger
                ec.stamp_triggers([(b + 1) * 4 for b in decimals_to_binary(
                    [story_idx, q_idx], [n_bits_story, n_bits_question])])

                # Wait for audio to finish (text stays visible)
                while ec.current_time < trial_start_time + question_duration:
                    ec.check_force_quit()
                    ec.wait_secs(0.1)

                ec.stop()
                ec.trial_ok()

                # Keep question visible and wait for space to continue (experimenter control only)
                # Redraw the question text so it stays on screen
                ec.screen_text(f"Question {q_data['question_num']} of 5",
                              pos=[0, 0.8], units='norm', color='w', font_size=18)

                if options is None:
                    # Free response
                    ec.screen_text(q_data['question_text'], pos=[0, 0.4], units='norm',
                                  color='w', font_size=32, wrap=True)
                    # ec.screen_text("Answer out loud.",
                    #               pos=[0, -0.1], units='norm', color='yellow', font_size=22)
                else:
                    # Multiple choice
                    ec.screen_text(q_data['question_text'], pos=[0, 0.5], units='norm',
                                  color='w', font_size=28, wrap=True)
                    # ec.screen_text("Read your answer choice:",
                    #               pos=[0, 0.25], units='norm', color='yellow', font_size=22)

                    # Display answer options
                    y_start = 0.0
                    y_spacing = 0.15
                    for i, option_text in enumerate(options):
                        y_pos = y_start - (i * y_spacing)
                        ec.screen_text(option_text, pos=[0, y_pos], units='norm',
                                      color='w', font_size=24, wrap=True)

                ec.flip()

            # Wait for experimenter to press space
            ec.wait_one_press(max_wait=np.inf, live_keys=['space'])

            response = "verbal_response"

            # Store response
            story_responses[f"q{q_data['question_num']}_response"] = response

            # Write question data
            ec.write_data_line("question", {
                "story_id": story_id,
                "question_num": q_data['question_num'],
                "question_text": q_data['question_text'],
                "response": response
            })

        ec.wait_secs(0.5)

    # End of experiment
    ec.screen_prompt("Ya did it! \n\nThank you for participating!",
                    live_keys=['space'])

print("\nExperiment completed successfully!")
