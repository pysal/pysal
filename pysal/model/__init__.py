import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "access",
        "spatialml",
        "mgwr",
        "spglm",
        "spint",
        "spopt",
        "spreg",
        "tobler",
    ],
)
