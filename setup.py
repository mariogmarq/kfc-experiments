from setuptools import find_packages, setup

requires = ["flex", "flexBlock", "flexclash"]

setup(
        name="blockExperiments",
        version="1.0.0",
        author="Garcia Marquez Mario and Rodriguez Barroso Nuria",
        keywords="FL federated-learning flexible blockchain experiments flexBLock",
        packages=find_packages(),
        install_requires=requires,
        python_requires=">=3.8.10",
)