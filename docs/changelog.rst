.. _changelog:

Changelog
=========

Version 0.2.1
-------------
**New Features:**
- **Schneider2000 Density Conversion**: Advanced 44-segment piecewise model for HU-to-density conversion
  
  - ``hu_to_density_schneider()``: Interpolated conversion using all 44 tissue segments
  - ``hu_to_density_schneider_piecewise()``: Exact piecewise conversion matching lookup table  
  - ``get_schneider_tissue_info()``: Lookup tissue information for specific HU values
  - ``compare_density_methods()``: Compare bilinear vs Schneider methods
  
- **Enhanced Accuracy**: Schneider model provides clinically validated densities with ~0.17-0.19 g/cm³ improved accuracy over bilinear model
- **Comprehensive Tissue Support**: Covers air, lung variations, soft tissues, bones, and metal implants
- **New Example**: ``06_schneider_density_conversion.py`` demonstrates advanced density conversion with visualizations
- **Comprehensive Test Suite**: 16 new test cases for Schneider functionality

**Improvements:**
- Extended documentation with density conversion methods comparison
- Updated CLAUDE.md with Schneider model guidance
- Enhanced attenuation conversion utilities with clinical-grade accuracy

Version 0.2.0
-------------
**Breaking Changes:**
- Modified config file loading mechanism in SimulationConfig class
- Updated API for configuration initialization

**New Features:**
- Comprehensive test suite with unit and integration tests
- Enhanced documentation with ReadTheDocs support
- GitHub Actions CI/CD pipeline
- Auto-generated API documentation
- Professional documentation structure

**Improvements:**
- Better test coverage for all components
- Improved code quality with automated checks
- Streamlined README for better user experience

Version 0.1.1
-------------
- Bug fixes and minor improvements from initial release

Version 0.1.0
-------------
- Initial release with core functionalities and examples
