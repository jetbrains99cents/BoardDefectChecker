# language.py
import json
import os
from PySide6.QtCore import Signal, QObject  # Import QObject and Signal

class SimpleLanguageManager(QObject):  # Inherit from QObject to use signals
    language_changed = Signal(str)  # Signal emitted when language changes

    def __init__(self, json_path="resources/language.json"):
        super().__init__()  # Initialize QObject
        self.json_path = json_path
        self.languages_dict = {}
        self.current_tab = None
        self.current_language = None

        if os.path.exists(self.json_path):
            print(f"[Debug] Loading language file from: {self.json_path}")
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.languages_dict = json.load(f)
                print(f"[Debug] Loaded dictionary keys: {list(self.languages_dict.keys())}")
            except json.JSONDecodeError as e:
                print(f"[Error] JSON decode error in {self.json_path}: {e}")
        else:
            print(f"Warning: {self.json_path} not found.")

    def set_tab(self, tab_name: str):
        print(f"[Debug] Calling set_tab('{tab_name}')")
        if tab_name in self.languages_dict:
            self.current_tab = tab_name
            print(f"[Debug] Tab set to: {self.current_tab}")
        else:
            print(f"Tab '{tab_name}' not found in the language file.")
            self.current_tab = None

    def set_language(self, language_code: str):
        if self.current_tab is None:
            print("No tab is set.")
            return
        print(f"[Debug] Attempting to set language to '{language_code}' for tab '{self.current_tab}'")
        if language_code in self.languages_dict.get(self.current_tab, {}):
            self.current_language = language_code
            print(f"Switched language to: {language_code} for tab {self.current_tab}")
            self.language_changed.emit(language_code)  # Emit signal with new language
        else:
            print(f"Language '{language_code}' not found for tab '{self.current_tab}'.")
            self.current_language = None

    def get_text(self, key: str) -> str:
        if not self.current_tab or not self.current_language:
            return key
        return self.languages_dict.get(self.current_tab, {}).get(self.current_language, {}).get(key, key)