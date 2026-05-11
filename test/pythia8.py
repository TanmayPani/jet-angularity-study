import marimo

__generated_with = "0.19.11"
app = marimo.App(width="columns")

with app.setup:
    import pythia8mc as pythia


@app.function
def setup():
    _pythia = pythia.Pythia()

    _pythia.readString("Print:quiet = on")
    _pythia.readString("Beams:idA = 2212")
    _pythia.readString("Beams:idB = 2212")
    _pythia.readString("Beams:eCM = 200.")
    _pythia.readString("HardQCD:all = on")

    _pythia.readString("PhaseSpace:pTHatMin = 11.0")
    _pythia.readString("PhaseSpace:pTHatMax = -1")

    _pythia.readString("PhaseSpace:bias2Selection = on")
    _pythia.readString("PhaseSpace:bias2SelectionPow = 4")
    _pythia.readString("PhaseSpace:bias2SelectionRef = 11.")

    # Turn on some kinematic weighting.
    _pythia.readString(
        "VariationFrag:List = {kineVar0 frag:aLund=0.6"
        + " frag:bLund=0.9 frag:rFactC=1.2"
        + " frag:rFactB=1.0 frag:ptSigma=0.4}"
    )
    _pythia.readString(
        "VariationFrag:List += {kineVar1 frag:aLund=0.6"
        + " frag:bLund=0.9 frag:rFactC=1.2"
        + " frag:rFactB=1.0 frag:ptSigma=0.3}"
    )
    _pythia.init()
    return _pythia


@app.function
def generate(pythia_gen, n_events, batch_size=-1, error_mode="none"):
    batch_size = n_events if batch_size < 0 else batch_size
    n_events_to_generate = n_events
    i_batch = 0
    while n_events_to_generate > 0:
        batch_size = (
            n_events_to_generate if batch_size > n_events_to_generate else batch_size
        )
        n_events_to_generate -= batch_size
        i_batch += 1
        print(f"{i_batch} batches generated...")
        yield pythia_gen.nextBatch(batch_size, error_mode), pythia_gen.infoPython()


@app.cell
def _():
    _pythia = setup()
    _pythia_batch_gen = generate(_pythia, 100)
    _pythia_batch, _pythia_info = next(_pythia_batch_gen)
    for _field in _pythia_batch.info.fields:
        if _field == "weights":
            _awk_batch_info = (
                _pythia_batch.info[_field][-1][0]
                if len(_pythia_batch.info[_field][-1]) > 0
                else 0.0
            )
            _pythia_info_last = getattr(_pythia_info, "weight")()
        else:
            _awk_batch_info = _pythia_batch.info[_field][-1]
            _pythia_info_last = getattr(_pythia_info, _field)()

        print(_field, _awk_batch_info, _pythia_info_last)
        assert _awk_batch_info == _pythia_info_last, f"{_field} failed!"
        print(_field, "passed...")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
