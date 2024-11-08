GATree
======

GATree is a Python library designed for implementing evolutionary decision trees using a standard genetic algorithm approach. The library provides functionalities for selection, mutation, and crossover operations within the decision tree structure, allowing users to evolve and optimise decision trees for various classification and clustering tasks. 🌲🧬

The library's core objective is to empower users in creating and fine-tuning decision trees through an evolutionary process, opening avenues for innovative approaches to classification and clustering problems. GATree enables the dynamic growth and adaptation of decision trees, offering a flexible and powerful tool for machine learning enthusiasts and practitioners. 🚀🌿

GATree is currently limited to classification and clustering tasks, with support for regression tasks planned for future releases. 💡

* **Free software:** MIT license
* **GitHub**: https://github.com/lahovniktadej/gatree
* **Python**: 3.9, 3.10, 3.11, 3.12
* **Operating systems**: Windows, Ubuntu, macOS

Genetic Operators in GATree
----------------------------------

The genetic algorithm for decision trees in GATree involves several key operators: *selection*, *elitism*, *crossover*, and *mutation*. Each of these operators plays a crucial role in the evolution and optimisation of the decision trees. Below is a detailed description of each operator within the context of the GATree class. 🧬

Selection
~~~~~~~~~

Selection is the process of choosing parent trees from the current population to produce offspring for the next generation. By default, the GATree class uses tournament selection, a method where a subset of the population is randomly chosen, and the best individual from this subset is selected.

Elitism
~~~~~~~

Elitism ensures that the best-performing individuals (trees) from the current generation are carried over to the next generation without any modification. This guarantees that the quality of the population does not decrease from one generation to the next.

Crossover
~~~~~~~~~

Crossover is a genetic operator used to combine the genetic information of two parent trees to generate new offspring. This enables exploration, which helps in creating diversity in the population and combining good traits from both parents.

Mutation
~~~~~~~~

Mutation introduces random changes to a tree to maintain genetic diversity and explore new solutions. This helps in avoiding local optima by introducing new genetic structures.

Documentation
-------------

The documentation is organised into the following sections:

* :ref:`user`
* :ref:`dev`
* :ref:`gatree`
* :ref:`about`

..  _user:

..  toctree::
    :maxdepth: 1
    :caption: User documentation

    user/usage

.. _dev:

.. toctree::
   :maxdepth: 1
   :caption: Developer documentation

   dev/installation
   dev/documentation

.. _gatree:

.. toctree::
   :maxdepth: 2
   :caption: GATree

   gatree/index

.. _about:

.. toctree::
   :maxdepth: 1
   :caption: About

   about/license
   about/code_of_conduct
