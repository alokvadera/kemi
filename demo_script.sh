#!/bin/bash
# Demo script for asciinema

python3 << 'EOF'
import sys
import time

# Simulate typing delay
def type_print(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

print("$ pip install kemi[local]")
time.sleep(0.5)
print("Collecting kemi[local]...")
time.sleep(0.3)
print("Installing: kemi-0.1.7")
time.sleep(0.2)
print()
time.sleep(0.3)

print("$ python3")
time.sleep(0.3)
print("Python 3.14.3 (v3.14.3:...) on darwin")
print('Type "help", "copyright", "credits" or "license" for more information.')
print()

type_print(">>> from kemi import Memory")
time.sleep(0.2)

type_print(">>> memory = Memory()")
time.sleep(1)

type_print(">>> memory.remember(\"alok\", \"I am vegetarian\")")
time.sleep(1)
print("'e3f2a1b8-4c9d-4e5f-a6b7-c8d9e0f1a2b3'")
time.sleep(0.3)

type_print(">>> memory.remember(\"alok\", \"I prefer dark mode\")")
time.sleep(1)
print("'a1b2c3d4-5e6f-7a8b-9c0d-e1f2a3b4c5d6'")
time.sleep(0.3)

type_print(">>> memory.remember(\"alok\", \"I live in Mumbai\")")
time.sleep(1)
print("'7f8e9a0b-1c2d-3e4f-5a6b-7c8d9e0f1a2b'")
time.sleep(0.3)

type_print(">>> results = memory.recall(\"alok\", \"what do I like?\")")
time.sleep(1.5)

type_print(">>> for r in results:")
type_print("...     print(r.content)")
time.sleep(0.5)
print("I prefer dark mode")
print("I live in Mumbai")
print("I am vegetarian")
time.sleep(0.5)

type_print(">>> memory.context_block(\"alok\", \"user preferences\")")
time.sleep(1)
print("'Relevant context from memory:\\n- I prefer dark mode\\n- I live in Mumbai\\n- I am vegetarian'")
time.sleep(0.5)

print()
type_print(">>> ")
EOF