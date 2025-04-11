# PyCodeCleanse

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

A lightweight, efficient tool to strip comments and docstrings from Python source code while maintaining full functionality. Perfect for code obfuscation, reducing file size, or preparing code for production deployment.

## Features

- Removes single-line comments (`#`)
- Eliminates multi-line comments and docstrings (`'''` and `"""`)
- Preserves string literals that use triple quotes
- Maintains code structure and indentation
- Works with Python 3.6+
- Simple command-line interface
- Can be imported as a module in other projects

## Installation

```bash
# Clone the repository
git clone https://github.com/SakibAhmedShuva/PyCodeCleanse.git

# Navigate to the directory
cd PyCodeCleanse
```

## Usage

### Command Line

```bash
python pycodeceanse.py input_file.py [output_file.py]
```

If no output file is specified, the cleaned code will be saved to `[original_name]_clean.py`.

### As a Module

```python
from pycodecleanse import remove_comments

# Clean a file
cleaned_file_path = remove_comments("my_script.py", "clean_script.py")

# Or use the default output naming
cleaned_file_path = remove_comments("my_script.py")
```

## Examples

### Before Cleaning

```python
# This is a comment
def hello_world():
    """
    This is a docstring that will be removed
    """
    # Another comment
    print("Hello, World!")  # End of line comment

'''
Multi-line comment
that spans several lines
'''
```

### After Cleaning

```python
def hello_world():
    print("Hello, World!")
```

## Use Cases

- Reduce file size for deployment
- Obfuscate code to protect intellectual property
- Prepare code for minification
- Clean up code before sharing publicly
- Improve readability for automated code analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

[Sakib Ahmed Shuva](https://github.com/SakibAhmedShuva)
