from setuptools import setup, find_packages

setup(
    name="mp4_to_jpg",
    version="1.0.3",
    author="Tadeas Fort",
    author_email="taddy.fort@gmail.com",
    description="A tool to extract frames from MP4 videos and remove duplicates",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tadeasf/mp4-to-jpg_click",
    packages=find_packages(),
    install_requires=[
        "click",
        "opencv-python",
        "tk",
        "tqdm",
        "loguru",
        "imagededup",
    ],
    entry_points={
        "console_scripts": [
            "mp4-to-jpg = mp4_to_jpg.mp4_to_jpg:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
