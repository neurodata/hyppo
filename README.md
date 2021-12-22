# hyppo
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[all-contrib]: https://img.shields.io/badge/all_contributors-23-orange.svg?style=flat 'All Contributors'
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![CircleCI](https://img.shields.io/circleci/build/github/neurodata/hyppo/main?style=flat)](https://app.circleci.com/pipelines/github/neurodata/hyppo?branch=main)
[![Codecov](https://img.shields.io/codecov/c/github/neurodata/hyppo?style=flat)](https://codecov.io/gh/neurodata/hyppo)
[![Netlify](https://img.shields.io/netlify/e5242ebd-631e-4330-b43e-85e428dac66a?style=flat)](https://app.netlify.com/sites/hyppo/deploys)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hyppo?style=flat)](https://pypi.org/project/hyppo/)
[![PyPI](https://img.shields.io/pypi/v/hyppo?style=flat)](https://pypi.org/project/hyppo/)
[![arXivshield](https://img.shields.io/badge/arXiv-1907.02088-red.svg?style=flat)](https://arxiv.org/abs/1907.02088)
[![All Contributors][all-contrib]](#contributors)

hyppo (**HYP**othesis Testing in **P**yth**O**n, pronounced "Hippo") is an open-source software package for multivariate hypothesis testing. We decided to develop hyppo for the following reasons:

* With the increase in the amount of data in many fields, hypothesis testing for high dimensional and nonlinear data is important
* Libraries in R exist, but their interfaces are inconsistent and most are not available in Python

hyppo intends to be a comprehensive multivariate hypothesis testing package and runs on all major versions of operating systems. It also includes novel tests not found in other packages. It is quick to install and free of charge. If you need to use multivariate hypothesis testing, be sure to give hyppo a try!

Website: [https://hyppo.neurodata.io/](https://hyppo.neurodata.io/)

## Installation

### Dependencies

hyppo requires the following:

* [python](https://www.python.org/) (>= 3.6)
* [numba](https://numba.pydata.org/) (>= 0.46)
* [numpy](https://numpy.org/)  (>= 1.17)
* [scipy](https://docs.scipy.org/doc/scipy/reference/) (>= 1.4.0)
* [scikit-learn](https://scikit-learn.org/stable/) (>= 0.22)
* [joblib](https://joblib.readthedocs.io/en/latest/) (>= 0.17.0)

### User installation

The easiest way to install hyppo is using `pip`

```sh
pip install hyppo
```

To upgrade to a newer release use the `--upgrade` flag

```sh
pip install --upgrade hyppo
```

The documentation includes more detailed [installation instructions](https://hyppo.neurodata.io/get_start/install.html).
hyppo is free software; you can redistribute it and/or modify it under the
terms of the [license](https://hyppo.neurodata.io/development/license.html).

## Release Notes

See the [release notes](https://hyppo.neurodata.io/changelog/index.html)
for a history of notable changes to hyppo.

## Development

We welcome new contributors of all experience levels. The hyppo
community goals are to be helpful, welcoming, and effective. The
[contributor guide](https://hyppo.neurodata.io/development/contributing.html)
has detailed information about contributing code, documentation and tests.

* Official source code: [https://github.com/neurodata/hyppo/tree/main/hyppo](https://github.com/neurodata/hyppo/tree/main/hyppo)
* Download releases: [https://pypi.org/project/hyppo/](https://pypi.org/project/hyppo/)
* Issue tracker: [https://github.com/neurodata/hyppo/issues](https://github.com/neurodata/hyppo/issues)

**Note: We have recently moved our benchmarks (with relevant figure replication code for our papers) folder to a new [repo](https://github.com/neurodata/hyppo-papers)!** We aim to add test development code and paper figure replication code there, and will add relevant tests (with tutorials) to hyppo.

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://sampan.me"><img src="https://avatars.githubusercontent.com/u/36676569?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sambit Panda</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/issues?q=author%3Asampan501" title="Bug reports">🐛</a> <a href="https://github.com/neurodata/hyppo/commits?author=sampan501" title="Code">💻</a> <a href="https://github.com/neurodata/hyppo/commits?author=sampan501" title="Documentation">📖</a> <a href="#ideas-sampan501" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-sampan501" title="Maintenance">🚧</a> <a href="https://github.com/neurodata/hyppo/pulls?q=is%3Apr+reviewed-by%3Asampan501" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/cshen6"><img src="https://avatars.githubusercontent.com/u/11924683?v=4?s=100" width="100px;" alt=""/><br /><sub><b>cshen6</b></sub></a><br /><a href="#ideas-cshen6" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="http://neurodata.io"><img src="https://avatars.githubusercontent.com/u/41842?v=4?s=100" width="100px;" alt=""/><br /><sub><b>joshua vogelstein</b></sub></a><br /><a href="#fundingFinding-jovo" title="Funding Finding">🔍</a> <a href="#mentoring-jovo" title="Mentoring">🧑‍🏫</a> <a href="#ideas-jovo" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://in.linkedin.com/in/satishpalaniappan/en"><img src="https://avatars.githubusercontent.com/u/10278507?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Satish Palaniappan</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=tpsatish95" title="Code">💻</a> <a href="#ideas-tpsatish95" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/junhaobearxiong"><img src="https://avatars.githubusercontent.com/u/30681308?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Junhao Xiong</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=junhaobearxiong" title="Code">💻</a> <a href="#ideas-junhaobearxiong" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="http://ericwb.me"><img src="https://avatars.githubusercontent.com/u/8883547?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Eric Bridgeford</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/issues?q=author%3Aebridge2" title="Bug reports">🐛</a> <a href="#ideas-ebridge2" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/ronakdm"><img src="https://avatars.githubusercontent.com/u/72722942?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ronak D. Mehta</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=ronakdm" title="Code">💻</a> <a href="#ideas-ronakdm" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/j1c"><img src="https://avatars.githubusercontent.com/u/5142539?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jaewon Chung</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/issues?q=author%3Aj1c" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/jdey4"><img src="https://avatars.githubusercontent.com/u/52499217?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jayanta Dey</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=jdey4" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/bvarjavand"><img src="https://avatars.githubusercontent.com/u/8294669?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Bijan Varjavand</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=bvarjavand" title="Code">💻</a></td>
    <td align="center"><a href="https://bdpedigo.github.io/"><img src="https://avatars.githubusercontent.com/u/25714207?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Benjamin Pedigo</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/issues?q=author%3Abdpedigo" title="Bug reports">🐛</a> <a href="https://github.com/neurodata/hyppo/commits?author=bdpedigo" title="Code">💻</a></td>
    <td align="center"><a href="http://alyakin314.github.io"><img src="https://avatars.githubusercontent.com/u/25692376?v=4?s=100" width="100px;" alt=""/><br /><sub><b>alyakin314</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=alyakin314" title="Code">💻</a> <a href="#ideas-alyakin314" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://vivekg.dev"><img src="https://avatars.githubusercontent.com/u/29757116?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Vivek Gopalakrishnan</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=v715" title="Code">💻</a></td>
    <td align="center"><a href="https://rflperry.github.io/"><img src="https://avatars.githubusercontent.com/u/13107341?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ronan Perry</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/issues?q=author%3Arflperry" title="Bug reports">🐛</a> <a href="https://github.com/neurodata/hyppo/commits?author=rflperry" title="Code">💻</a> <a href="#ideas-rflperry" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/neurodata/hyppo/pulls?q=is%3Apr+reviewed-by%3Arflperry" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/neurodata/hyppo/commits?author=rflperry" title="Documentation">📖</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/PSSF23"><img src="https://avatars.githubusercontent.com/u/20309845?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Haoyin Xu</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=PSSF23" title="Code">💻</a> <a href="https://github.com/neurodata/hyppo/commits?author=PSSF23" title="Documentation">📖</a> <a href="https://github.com/neurodata/hyppo/pulls?q=is%3Apr+reviewed-by%3APSSF23" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/hadasarik"><img src="https://avatars.githubusercontent.com/u/39025628?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Hadas Arik</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=hadasarik" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/kareef928"><img src="https://avatars.githubusercontent.com/u/51966539?v=4?s=100" width="100px;" alt=""/><br /><sub><b>kareef928</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=kareef928" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Verathagnus"><img src="https://avatars.githubusercontent.com/u/59093644?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Verathagnus</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=Verathagnus" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/dlee0156"><img src="https://avatars.githubusercontent.com/u/47963020?v=4?s=100" width="100px;" alt=""/><br /><sub><b>dlee0156</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=dlee0156" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/najmieh"><img src="https://avatars.githubusercontent.com/u/31998054?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Najmieh Sadat Safarabadi</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=najmieh" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/TacticalFallacy"><img src="https://avatars.githubusercontent.com/u/56208921?v=4?s=100" width="100px;" alt=""/><br /><sub><b>TacticalFallacy</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=TacticalFallacy" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/darsh-patel"><img src="https://avatars.githubusercontent.com/u/70541374?v=4?s=100" width="100px;" alt=""/><br /><sub><b>darsh-patel</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=darsh-patel" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/zdbzdb123123"><img src="https://avatars.githubusercontent.com/u/95720890?v=4?s=100" width="100px;" alt=""/><br /><sub><b>zdbzdb123123</b></sub></a><br /><a href="https://github.com/neurodata/hyppo/commits?author=zdbzdb123123" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://allcontributors.org) specification.
Contributions of any kind are welcome!

## Project History

hyppo is a rebranding of mgcpy, which was founded in Novemember 2018.
mgcpy was designed and written by Satish Palaniappan, Sambit
Panda, Junhao Xiong, Sandhya Ramachandran, and Ronak Mehtra. hyppo
was designed and written by Sambit Panda and first released February 2020.
