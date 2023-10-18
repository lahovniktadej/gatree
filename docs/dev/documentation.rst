Documentation
=============

To locally generate and preview documentation, run the following commands in the root directory of the project:

.. code:: sh

    poetry install --extras docs
    poetry run sphinx-build ./docs ./docs/build

If the documentation builds successfully, it can be previewed in the ``docs/build`` folder by opening the ``index.html`` file.