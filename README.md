# drain-rs

Drain provides a machinism for online log categorization.

The goal of this particular project is to provide a nice, fast, rust upgrade to the original [drain](https://github.com/logpai/logparser/tree/master/logparser/Drain) implementation.
Original paper here:
- Pinjia He, Jieming Zhu, Zibin Zheng, and Michael R. Lyu. [Drain: An Online Log Parsing Approach with Fixed Depth Tree](http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf), Proceedings of the 24th International Conference on Web Services (ICWS), 2017.

- [x] Implement basic algorithm
- [x] Utilize GROK instead of vanilla regex for template creation (allows type inferrence, better patterns). Along with supporting GROK, the ability to add custom patterns would be nice.
- [x] Add ability to set Overall log template. Some logs have a well known format and auto parsing is not particularly useful for known formats. But, usually, known formats have free text fields, and those would benefit from some auto parsing
- [ ] Decouple command line utility from drain implementation
- [x] ability to save and read in state

This is a WIP, 0.0.x
