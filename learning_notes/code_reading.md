In run_gemini.py, the managerAgent.run(task, images=compressed_problem_images) is the main function that starts the codeact loop.

- task is the prompt and the compressed_problem_images are the images in the problem.

- Input of the ask_image_expert tool is the image reference (placeholder like <image_10>) and a question. Even the images are input to the managerAgent, the ask_image_expert tool is still needed to get the answer from the image, which can obtain more detailed information from the image.