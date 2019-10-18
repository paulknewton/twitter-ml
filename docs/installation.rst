.. highlight:: shell

============
Installation
============

Dependencies
------------

* `Apache Spark <y>`_ - You will need spark installed/downloaded on your machine to run this. Note: when running on a Mac, I had to (downgrade) to Java v8 to get Spark to run (otherwise even the Spark examples fail).
* `Apache Kafka <https://kafka.apache.org/>`_ - Our Twitter feed uses Apache Kafka to publish the stream of tweets. Download it and unpack the tarball.
* The scipy toolkit is installed automatically below, but on Windows you need to make sure you have the Microsoft C++ libraries installed. Follow the instructions `here <https://www.microsoft.com/en-us/download/details.aspx?id=48145>`_.

Stable release
--------------

To install Twitter ML, run this command in your terminal:

.. code-block:: console

    $ pip install twitter_ml

This is the preferred method to install Twitter ML, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Twitter ML can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/paulknewton/twitter_ml

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/paulknewton/twitter_ml/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/paulknewton/twitter_ml
.. _tarball: https://github.com/paulknewton/twitter_ml/tarball/master
