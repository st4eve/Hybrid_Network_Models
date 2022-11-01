"""Hybrid Network Models 2022"""

from PyInquirer import prompt

from Plot import *


def command(cmd_type, message, choices=None, predefined_output=None):
    """Utility command management function"""
    if predefined_output is None:
        name = "variable"
        question = [
            {"type": cmd_type, "name": name, "message": message, "choices": choices}
        ]
        if cmd_type == "input":
            question = [{"type": cmd_type, "name": name, "message": message}]
        elif cmd_type == "list":
            question = [
                {"type": cmd_type, "name": name, "message": message, "choices": choices}
            ]
        answer = prompt(question)
        return answer[name]
    else:
        return predefined_output


def cli_plot(plotter):
    """Manages the plotting given a certain plot"""
    plotter.plot()

    while True:
        options = plotter.get_options()
        option_strings = []
        option_string_map = {}
        for key, val in options.items():
            option_string = key + " = " + str(val)
            option_strings.append(option_string)
            option_string_map[option_string] = key

        option_strings.append("save")
        option_strings.append("exit")
        option_strings.insert(0, "plot")
        option_strings.append("settings from file")
        option_strings.append("new plot")

        cmd_type = "list"
        message = "Please select plotting option to modify, or select plot, save, settings from file, or exit"
        choices = option_strings
        option = command(cmd_type, message, choices)

        if option == "save":
            cmd_type = "input"
            message = "Enter name of folder to save to:"
            save_name = command(cmd_type, message)
            plotter.save(save_name)
            plotter.close()
        elif option == "plot":
            plotter.close()
            plotter.plot()
        elif option == "exit":
            exit()
        elif option == "settings from file":
            cmd_type = "input"
            message = "Enter path to saved settings"
            path = command(cmd_type, message)
            plotter.apply_saved_settings(path)
        elif option == "new plot":
            return
        else:
            cmd_type = "input"
            message = "Please enter the new value for " + option + ":"
            value = command(cmd_type, message, choices)
            plotter.set_option(option_string_map[option], value)
