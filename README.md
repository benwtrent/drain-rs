# drain-rs

Drain provides a machinism for online log categorization.

This version provides:

- serialization/deserialization of drain state via serde json
- support for GROK patterns for more accurate categories and variable filtering

The goal of this particular project is to provide a nice, fast, rust upgrade to the original [drain](https://github.com/logpai/logparser/tree/master/logparser/Drain) implementation.
Original paper here:
- Pinjia He, Jieming Zhu, Zibin Zheng, and Michael R. Lyu. [Drain: An Online Log Parsing Approach with Fixed Depth Tree](http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf), Proceedings of the 24th International Conference on Web Services (ICWS), 2017.

  

This is a WIP, 0.0.x
