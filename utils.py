def format_names(latest_detection_data):
    names = {}
    for obj in latest_detection_data:
        name = obj["name"]
        if names.get(name) is None:
            names[name] = 1
        else:
            names[name] += 1
    return names
