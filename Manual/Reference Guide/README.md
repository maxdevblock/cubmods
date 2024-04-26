The Reference Guide is organised in package modules.

For each module, are listed: Functions (main and ancillary) and Classes.

***

In the functions' titles are listed required `args` only; default optional `kwargs` are explained in the _Arguments_ list with the default value. Example

```Python
from cubmods import modulename
```

## `.function(arg1, arg2)`

Description of the function.

- Arguments
  - `arg1` (_type_): description
  - `arg2` (_type_): description
  - `kwarg1=<default>` (_type_): description; `<default>` is the default value of the `kwarg`
  - `kwarg2=<default>` (_type_): description; `<default>` is the default value of the `kwarg`
- Returns
  - Description of returned values.

***

In Classes title no `arg` nor `kwarg` are listed; methods and functions are explained in the _Methods_ and _Functions_ lists. Example

```Python
from cubmods import modulename
```

## `SomeClass`

Description of the Class.

- Methods
  - `.method1` (_type_): description of what the method returns; for example, if `drawn` is an instance of `CUBsample` returned by a `cub.draw(*args,**kwargs)` function, calling the method `drawn.rv` will return the _array_ of the drawn sample
  - `.method2` (_type_): description of what the method returns...
- Functions
  - `.function1(arg1, arg2)`

    Description of the function.

    - Arguments
      - `arg1` (_type_): description
      - `arg2` (_type_): description
      - `kwarg1=<default>` (_type_): description; `<default>` is the default value of the `kwarg`
      - `kwarg2=<default>` (_type_): description; `<default>` is the default value of the `kwarg`
    - Returns
      - Description of returned values.
  - `.function2(arg1, arg2)`

    Description of the function.

    - Arguments
      - `arg1` (_type_): description
      - `arg2` (_type_): description
      - `kwarg1=<default>` (_type_): description; `<default>` is the default value of the `kwarg`
      - `kwarg2=<default>` (_type_): description; `<default>` is the default value of the `kwarg`
    - Returns
      - Description of returned values.
  
