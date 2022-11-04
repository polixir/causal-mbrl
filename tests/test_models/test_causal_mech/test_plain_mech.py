from cmrl.models.causal_mech.plain_mech import PlainMech
from cmrl.types import Variable, ContinuousVariable, DiscreteVariable


def test_without_coder():
    node_dim = 1

    input_variables = [ContinuousVariable(name="state0", dim=node_dim), ContinuousVariable(name="action0", dim=node_dim)]
    output_variables = [ContinuousVariable(name="state0", dim=node_dim)]

    mech = PlainMech(
        input_variables=input_variables,
        output_variables=output_variables,
        node_dim=node_dim,
        variable_encoders={"state0": None, "action0": None},
        variable_decoders={"state0": None},
    )


def test_single_dim_continuous():
    input_variables = [ContinuousVariable(1), ContinuousVariable(1)]
    output_variables = [ContinuousVariable(1)]

    mech = PlainMech(input_variables=input_variables, output_variables=output_variables)
