#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRT Story Presentation Experiment Script

This script presents emotional prosody stories to participants and collects responses
to comprehension questions. Stories are played with audio, followed by question audio
and visual response options using mouse clicks.

@author: Tong
"""

import os
import numpy as np
import pandas as pd
from expyfun import ExperimentController, decimals_to_binary
from expyfun.visual import Circle, ProgressBar
from expyfun.io import read_wav
import threading

# %% Set parameters
fs = 48000  # Sample rate
n_channel = 2  # Stereo output
stim_db = 65  # Stimulus volume in dB

pause_dur = 1.0  # Pause duration between story and questions

# %% Load story and question data
# Set up paths
pilot_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
stim_path = pilot_path + "stim_normalized/"
csv_path = stim_path + "story_questions_mapping.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

# Load the CSV
df = pd.read_csv(csv_path)

# Get unique stories
stories_df = df.groupby('story_id').first().reset_index()
print(f"Found {len(stories_df)} stories to present")

# %% Preload all audio files
print("\nPreloading audio files...")

story_audio = {}
question_audio = {}

for idx, story_row in stories_df.iterrows():
    story_id = story_row['story_id']
    story_name = story_row['story_name']

    # Load story audio
    story_file = pilot_path + story_row['story_path']
    print(f"Loading story: {story_name} ({story_id})")
    try:
        temp, _ = read_wav(story_file)
        story_audio[story_id] = temp
        print(f"  ✓ Story loaded: {len(temp)/fs:.1f}s")
    except Exception as e:
        print(f"  ✗ Error loading story: {e}")
        story_audio[story_id] = None

    # Load question audio for this story
    story_questions = df[df['story_id'] == story_id]
    question_audio[story_id] = []

    for _, q_row in story_questions.iterrows():
        q_file = pilot_path + q_row['question_audio_path']
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

def create_response_circles(ec, n_options=4):
    """Create circles for multiple choice responses"""
    circles = []
    circle_radius = (0.03, 0.04)

    # Position circles horizontally
    if n_options == 4:
        x_positions = [-0.45, -0.15, 0.15, 0.45]
    else:
        # Distribute evenly
        spacing = 0.6 / (n_options - 1) if n_options > 1 else 0
        x_positions = [-0.3 + i * spacing for i in range(n_options)]

    for i, x_pos in enumerate(x_positions):
        circle = Circle(ec, radius=circle_radius, pos=(x_pos, -0.3),
                       units='norm', fill_color=None, line_color='white', line_width=3)
        circles.append(circle)

    return circles, x_positions

# %% Experiment instructions
instruction_text = """Welcome to the Prosody Recognition Task!

You will listen to several emotional stories.
After each story, you will answer questions about it.

For multiple choice questions:
- Click on the circle next to your answer (A, B, C, or D)

For open-ended questions:
- You will be asked to provide a verbal response

Press the SPACE bar to begin."""

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
n_bits_story = int(np.ceil(np.log2(len(stories_df))))
n_bits_question = int(np.ceil(np.log2(5)))  # Max 5 questions per story

with ExperimentController(**ec_args) as ec:
    # Show instructions
    ec.screen_prompt(instruction_text, live_keys=['space'])

    # Progress bar
    pb = ProgressBar(ec, [0, -.2, 1.5, .2], colors=('xkcd:teal blue', 'w'))

    # Preload first story
    first_story_id = stories_df.iloc[0]['story_id']
    print(f"\nPreloading first story: {first_story_id}")
    show_loading_screen(ec, "Preparing first story...")
    if story_audio[first_story_id] is not None:
        ec.load_buffer(story_audio[first_story_id])

    trial_start_time = -np.inf

    # Loop through each story
    for story_idx, story_row in stories_df.iterrows():
        story_id = story_row['story_id']
        story_name = story_row['story_name']
        emotion = story_row['emotion']

        print(f"\n{'='*60}")
        print(f"Story {story_idx + 1}/{len(stories_df)}: {story_name} ({emotion})")
        print(f"{'='*60}")

        # Display progress
        ec.screen_text(f"Story {story_idx + 1} of {len(stories_df)}: {story_name}",
                      pos=[0, 0.5], units='norm', color='w')
        ec.screen_text("Click anywhere to start the story",
                      pos=[0, 0], units='norm', color='w')
        pb.draw()
        ec.flip()

        # Wait for mouse click to start story
        ec.wait_one_click(max_wait=np.inf)

        # Play story
        if story_audio[story_id] is not None:
            story_duration = len(story_audio[story_id]) / fs

            ec.screen_text(f"Now playing: {story_name}", pos=[0, 0], units='norm', color='w')
            ec.flip()

            # Identify trial
            trial_id = f"{story_id}_story"
            ec.identify_trial(ec_id=trial_id, ttl_id=[])

            # Start playback
            trial_start_time = ec.start_stimulus()
            ec.wait_secs(0.1)

            # Trigger
            ec.stamp_triggers([(b + 1) * 4 for b in decimals_to_binary([story_idx], [n_bits_story])])

            # Wait for story to finish
            while ec.current_time < trial_start_time + story_duration:
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

        # Present questions for this story
        questions = question_audio[story_id]
        story_responses = {}

        for q_idx, q_data in enumerate(questions):
            print(f"\n  Question {q_data['question_num']}: {q_data['question_text'][:50]}...")

            # Display prompt to click to hear question
            ec.screen_text(f"Question {q_data['question_num']} of 5",
                          pos=[0, 0.3], units='norm', color='w')
            ec.screen_text("Click to hear the question",
                          pos=[0, 0], units='norm', color='w')
            ec.flip()

            # Wait for click
            ec.wait_one_click(max_wait=np.inf)

            # Load and play question audio
            if q_data['audio'] is not None:
                ec.load_buffer(q_data['audio'])
                question_duration = len(q_data['audio']) / fs

                ec.screen_text(f"Question {q_data['question_num']}",
                              pos=[0, 0], units='norm', color='w')
                ec.flip()

                # Identify trial
                q_trial_id = f"{story_id}_q{q_data['question_num']}"
                ec.identify_trial(ec_id=q_trial_id, ttl_id=[])

                # Play question
                trial_start_time = ec.start_stimulus()
                ec.wait_secs(0.1)

                # Trigger
                ec.stamp_triggers([(b + 1) * 4 for b in decimals_to_binary(
                    [story_idx, q_idx], [n_bits_story, n_bits_question])])

                # Wait for question to finish
                while ec.current_time < trial_start_time + question_duration:
                    ec.check_force_quit()
                    ec.wait_secs(0.1)

                ec.stop()
                ec.trial_ok()

            # Show answer options
            options = parse_answer_options(q_data['answer_options'])

            if options is None:
                # Free response
                ec.screen_text(q_data['question_text'], pos=[0, 0.3], units='norm', color='w')
                ec.screen_text("Please provide your verbal response.",
                              pos=[0, 0], units='norm', color='w')
                ec.screen_text("Click when you have finished responding.",
                              pos=[0, -0.3], units='norm', color='w')
                ec.flip()

                ec.wait_one_click(max_wait=np.inf)
                response = "verbal_response"
            else:
                # Multiple choice
                n_options = len(options)
                circles, x_positions = create_response_circles(ec, n_options)

                # Display question text and options
                ec.screen_text(q_data['question_text'], pos=[0, 0.5], units='norm',
                              color='w', font_size=24, wrap=True)

                # Draw circles and option labels
                for i, (circle, x_pos, option_text) in enumerate(zip(circles, x_positions, options)):
                    circle.draw()
                    ec.screen_text(option_text, pos=[x_pos, -0.45], units='norm',
                                  color='w', font_size=20, wrap=True)

                ec.flip()

                # Wait for click on circle
                click, ind = ec.wait_for_click_on(circles, max_wait=np.inf)
                response = options[ind] if ind is not None else "no_response"

                # Show feedback
                for i, (circle, x_pos, option_text) in enumerate(zip(circles, x_positions, options)):
                    if i == ind:
                        filled_circle = Circle(ec, radius=(0.03, 0.04), pos=(x_pos, -0.3),
                                              units='norm', fill_color='white', line_color='white', line_width=3)
                        filled_circle.draw()
                    else:
                        circle.draw()
                    ec.screen_text(option_text, pos=[x_pos, -0.45], units='norm',
                                  color='w', font_size=20, wrap=True)
                ec.flip()
                ec.wait_secs(0.5)

            # Store response
            story_responses[f"q{q_data['question_num']}_response"] = response

            # Write question data
            ec.write_data_line("question", {
                "story_id": story_id,
                "question_num": q_data['question_num'],
                "question_text": q_data['question_text'],
                "response": response
            })

        # Preload next story buffer if not the last one
        if story_idx < len(stories_df) - 1:
            next_story_id = stories_df.iloc[story_idx + 1]['story_id']
            if story_audio[next_story_id] is not None:
                show_loading_screen(ec, "Loading next story...")
                ec.load_buffer(story_audio[next_story_id])

        # Update progress bar
        pb.update_bar((story_idx + 1) / len(stories_df) * 100)
        pb.draw()
        ec.wait_secs(0.5)

    # End of experiment
    ec.screen_prompt("Thank you for participating!\n\nPress SPACE to exit.",
                    live_keys=['space'])

print("\nExperiment completed successfully!")
