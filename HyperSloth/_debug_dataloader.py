import os


def _debug_dataloader(dataloader, tokenizer, n_example=10):
    """
    Debug function to log samples from the training dataloader in an HTML format.
    Outputs to both terminal (with colors) and an HTML file with CSS styling.
    """
    from copy import deepcopy

    g = iter(dataloader)
    html_path = ".log/dataloader_examples.html"
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    # Create HTML file with CSS styling
    with open(html_path, "w") as html_file:
        html_file.write(
            """<!DOCTYPE html>
    <html>
    <head>
        <title>Dataloader Examples</title>
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        
        @media (prefers-color-scheme: light) {
            body { background-color: #ffffff; color: #333; }
            .trainable { background-color: #FFEBCD; color: #333; }
            .context { background-color: #E0FFE0; color: #333; }
            th { background-color: #f2f2f2; }
            th, td { border-color: #ddd; }
        }
        
        @media (prefers-color-scheme: dark) {
            body { background-color: #222; color: #f0f0f0; }
            .trainable { background-color: #664a20; color: #f0f0f0; }
            .context { background-color: #2a5a2a; color: #f0f0f0; }
            th { background-color: #444; color: #f0f0f0; }
            th, td { border-color: #555; }
        }
        
        .trainable, .context { padding: 2px; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid; padding: 8px; text-align: left; }
        h2 { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>Dataloader Examples</h1>
        <p>This file contains examples of training data with context and trainable parts.</p>
    """
        )

        for i in range(n_example):
            batch = next(g)
            input_ids = batch["input_ids"][0]
            label_ids = batch["labels"][0]
            parts_mask = label_ids >= 0  # True is trainable, False is context

            # Find split points where trainable/non-trainable sections change
            split_points = (
                [0]
                + [
                    i
                    for i, val in enumerate(parts_mask)
                    if i > 0 and val != parts_mask[i - 1]
                ]
                + [len(parts_mask)]
            )

            colored_parts = []
            html_file.write(f"\n    <h2>Example {i+1}</h2>\n")
            html_file.write(
                "    <table>\n        <tr><th>Text</th><th>Label</th></tr>\n"
            )

            for a, b in zip(split_points[:-1], split_points[1:]):
                text = tokenizer.decode(input_ids[a:b])
                is_trainable = parts_mask[a]

                # Colored text for terminal
                colored_text = (
                    f"\033[93m{text}\033[0m"
                    if is_trainable
                    else f"\033[92m{text}\033[0m"
                )
                colored_parts.append(colored_text)

                # HTML with CSS classes
                css_class = "trainable" if is_trainable else "context"
                label = "🟠 TRAIN" if is_trainable else "🟢 CONTEXT"

                # Escape HTML special characters
                text_escaped = (
                    text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                )

                # Add row to HTML table
                html_file.write(
                    f'        <tr>\n            <td><span class="{css_class}">{text_escaped}</span></td>\n'
                    f"            <td>{label}</td>\n        </tr>\n"
                )

            html_file.write("    </table>\n")

            # Colored text for terminal
            colored_output = "".join(colored_parts)
            terminal_msg = f"\n=== EXAMPLE #{i+1} ===\n" + colored_output + "\n"
            if i == 0:
                print(terminal_msg)

        html_file.write("</body>\n</html>")

    print(f"More training debug examples written to {html_path}")
