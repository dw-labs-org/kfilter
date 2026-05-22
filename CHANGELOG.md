# Changelog

## [0.5.0](https://github.com/dw-labs-org/kfilter/compare/v0.4.0...v0.5.0) (2026-05-22)


### ⚠ BREAKING CHANGES

* nonlinear measurements ([#64](https://github.com/dw-labs-org/kfilter/issues/64))
* add predict to system traits
* add linearisable system trait, split system trait
* Kalman trait Error type, make predict and update fallible

### Features

* add predict to system traits ([8ee8d30](https://github.com/dw-labs-org/kfilter/commit/8ee8d30654aca852fb8481f4b5e07035eb034444))
* Kalman trait Error type, make predict and update fallible ([1dfa09e](https://github.com/dw-labs-org/kfilter/commit/1dfa09e7c1059f4f1abedb26ec3735c0b4b59e77)), closes [#56](https://github.com/dw-labs-org/kfilter/issues/56)
* nalgebra defmt ([dbfa0cd](https://github.com/dw-labs-org/kfilter/commit/dbfa0cd42c4193104f75ff0289b009d879810151))
* nonlinear measurements ([#64](https://github.com/dw-labs-org/kfilter/issues/64)) ([5237932](https://github.com/dw-labs-org/kfilter/commit/52379324095830534e22115898fff4dd24abf2b1))


### Bug Fixes

* missing error on predict trait ([f1a9793](https://github.com/dw-labs-org/kfilter/commit/f1a97931856180627967e9b25f646ee39dd28304))
* missing error on predict trait ([dec582d](https://github.com/dw-labs-org/kfilter/commit/dec582d415c401123243cdee860a1c81889d1543))


### Miscellaneous Chores

* release 0.5.0 ([#67](https://github.com/dw-labs-org/kfilter/issues/67)) ([0a3a042](https://github.com/dw-labs-org/kfilter/commit/0a3a0428a7acabc2448caabc8205c019e1f1008d))


### Code Refactoring

* add linearisable system trait, split system trait ([7c10a11](https://github.com/dw-labs-org/kfilter/commit/7c10a111be207930a1b790848f1251a3284aed49))

## [0.4.0](https://github.com/dw-labs-org/kfilter/compare/v0.3.1...v0.4.0) (2025-04-30)


### ⚠ BREAKING CHANGES

* Normalise the covariance on update instead of unwrapping

### Features

* add simple defmt support to types ([21ce8f6](https://github.com/dw-labs-org/kfilter/commit/21ce8f692e84f14b91726fc2c2529266a58bcac7))
* add simple defmt support to types ([c1fd92c](https://github.com/dw-labs-org/kfilter/commit/c1fd92c54546fba95b0d4f7e265a2874f3042035))
* Normalise the covariance on update instead of unwrapping ([92c32dd](https://github.com/dw-labs-org/kfilter/commit/92c32dda68b1abbd28c633a4d84b9d38220f7887)), closes [#45](https://github.com/dw-labs-org/kfilter/issues/45)


### Bug Fixes

* no-std support with nalgebra + ci ([9170d1a](https://github.com/dw-labs-org/kfilter/commit/9170d1aa5b81c239c73a8524a33613d287a00e65)), closes [#43](https://github.com/dw-labs-org/kfilter/issues/43)
* remove serde support from types with functions. Add check script ([f48dd38](https://github.com/dw-labs-org/kfilter/commit/f48dd388a64b0099491c67132d8a9d9f363b1768))


### Miscellaneous Chores

* release 0.4.0 ([50e7bb4](https://github.com/dw-labs-org/kfilter/commit/50e7bb47bb017f0fcb78217c2024ca5cf9dcdf50))

## [0.3.1](https://github.com/dw-labs-org/kfilter/compare/v0.3.0...v0.3.1) (2024-08-19)


### Bug Fixes

* system impl for LinearNoInputSystem U=0 ([9545a2e](https://github.com/dw-labs-org/kfilter/commit/9545a2e75ad7de87ff8a8f4e6f707dd27f256a2e)), closes [#40](https://github.com/dw-labs-org/kfilter/issues/40)

## [0.3.0](https://github.com/dw-labs-org/kfilter/compare/v0.2.2...v0.3.0) (2024-07-30)


### Features

* serde Serialize and Deserialize feature for all structs that can ([1c7754c](https://github.com/dw-labs-org/kfilter/commit/1c7754c88d98a82a0f7791ad4eebbdf6e0214314))
* type aliases for Kalman linear and EKF ([#37](https://github.com/dw-labs-org/kfilter/issues/37)) ([f0d322a](https://github.com/dw-labs-org/kfilter/commit/f0d322a2ec77ca574cfd89ddb53836c4ff881172))
* type aliases for Kalman1M linear ([e6216b6](https://github.com/dw-labs-org/kfilter/commit/e6216b61690a5c77f3f0f1dae8be2e64d7d3124f))
* zero-order-hold transition matrix conversion ([e7db45f](https://github.com/dw-labs-org/kfilter/commit/e7db45f304199a703d2a3c468f74adf3ec8c7177))
