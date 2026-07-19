# Public Toolbox
#
# Shared across every character. Any character flagged "admin": true in
# their personality file can propose new functions to add here (subject to
# your approval each time). Once a function exists here, ANY character —
# admin or not — can call it freely, no confirmation needed, since it was
# already reviewed by you at write-time.
#
# Functions here are written by AI proposals you've approved. Available to
# that code at write-time: the helpers in toolbox_helpers.py (e.g.
# play_sound). No raw imports, no file/network access beyond what those
# helpers expose.
#
# This file starts empty on purpose.

def count():
    for i in range(10):
        print(i)