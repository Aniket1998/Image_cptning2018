       ЃK"	  РO%ЫжAbrain.Event:2ЏJ{)Ќ     5lІЛ	гЕПO%ЫжA"и

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

global_step
VariableV2*
shape: *
shared_name *
_class
loc:@global_step*
dtype0	*
	container *
_output_shapes
: 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_class
loc:@global_step*
validate_shape(*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
y
MatchingFiles/patternConst"/device:CPU:0*%
valueB Bdata/seq01.train.csv*
dtype0*
_output_shapes
: 
i
MatchingFilesMatchingFilesMatchingFiles/pattern"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ
`
compression_typeConst"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
^
buffer_sizeConst"/device:CPU:0*
valueB		 R*
dtype0	*
_output_shapes
: 
V
countConst"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
[

batch_sizeConst"/device:CPU:0*
value	B	 Rd*
dtype0	*
_output_shapes
: 
c
num_parallel_callsConst"/device:CPU:0*
dtype0*
value	B :*
_output_shapes
: 
X
count_1Const"/device:CPU:0*
dtype0	*
value	B	 R*
_output_shapes
: 
ђ
OneShotIteratorOneShotIterator"/device:CPU:0*0
dataset_factoryR
_make_dataset_IBZ7uUdCWTo*
output_types
2*
shared_name *9
output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
	container *
_output_shapes
: 
h
IteratorToStringHandleIteratorToStringHandleOneShotIterator"/device:CPU:0*
_output_shapes
: 
б
IteratorGetNextIteratorGetNextOneShotIterator"/device:CPU:0*
output_types
2*9
output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
G
ConstConst*
value	B :*
dtype0*
_output_shapes
: 
Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 

splitSplitsplit/split_dimIteratorGetNext*
	num_split*
T0*Ц
_output_shapesГ
А:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
N
	rnn/ShapeShapesplit*
T0*
out_type0*
_output_shapes
:
a
rnn/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
c
rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
c
rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
_output_shapes
: 
k
)rnn/BasicLSTMCellZeroState/ExpandDims/dimConst*
dtype0*
value	B : *
_output_shapes
: 
Ђ
%rnn/BasicLSTMCellZeroState/ExpandDims
ExpandDimsrnn/strided_slice)rnn/BasicLSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
j
 rnn/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
h
&rnn/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
и
!rnn/BasicLSTMCellZeroState/concatConcatV2%rnn/BasicLSTMCellZeroState/ExpandDims rnn/BasicLSTMCellZeroState/Const&rnn/BasicLSTMCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
k
&rnn/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
З
 rnn/BasicLSTMCellZeroState/zerosFill!rnn/BasicLSTMCellZeroState/concat&rnn/BasicLSTMCellZeroState/zeros/Const*

index_type0*
T0*'
_output_shapes
:џџџџџџџџџ
m
+rnn/BasicLSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
І
'rnn/BasicLSTMCellZeroState/ExpandDims_1
ExpandDimsrnn/strided_slice+rnn/BasicLSTMCellZeroState/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
m
+rnn/BasicLSTMCellZeroState/ExpandDims_2/dimConst*
dtype0*
value	B : *
_output_shapes
: 
І
'rnn/BasicLSTMCellZeroState/ExpandDims_2
ExpandDimsrnn/strided_slice+rnn/BasicLSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
j
(rnn/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
р
#rnn/BasicLSTMCellZeroState/concat_1ConcatV2'rnn/BasicLSTMCellZeroState/ExpandDims_2"rnn/BasicLSTMCellZeroState/Const_2(rnn/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*

Tidx0*
_output_shapes
:
m
(rnn/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
"rnn/BasicLSTMCellZeroState/zeros_1Fill#rnn/BasicLSTMCellZeroState/concat_1(rnn/BasicLSTMCellZeroState/zeros_1/Const*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
m
+rnn/BasicLSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
І
'rnn/BasicLSTMCellZeroState/ExpandDims_3
ExpandDimsrnn/strided_slice+rnn/BasicLSTMCellZeroState/ExpandDims_3/dim*

Tdim0*
T0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:
Л
;rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"   P   *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:
­
9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *<yО*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
­
9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *<y>*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 

Crnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform;rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
seed2.*
seedвЎК	*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes

:P

9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSub9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/max9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes
: 

9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulCrnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P

5rnn/basic_lstm_cell/kernel/Initializer/random_uniformAdd9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mul9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
Н
rnn/basic_lstm_cell/kernel
VariableV2*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
	container *
shape
:P*
shared_name *
_output_shapes

:P
џ
!rnn/basic_lstm_cell/kernel/AssignAssignrnn/basic_lstm_cell/kernel5rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
use_locking(*
_output_shapes

:P
p
rnn/basic_lstm_cell/kernel/readIdentityrnn/basic_lstm_cell/kernel*
T0*
_output_shapes

:P
Є
*rnn/basic_lstm_cell/bias/Initializer/zerosConst*
valueBP*    *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
:P
Б
rnn/basic_lstm_cell/bias
VariableV2*
	container *
shape:P*
shared_name *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
:P
ъ
rnn/basic_lstm_cell/bias/AssignAssignrnn/basic_lstm_cell/bias*rnn/basic_lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:P
h
rnn/basic_lstm_cell/bias/readIdentityrnn/basic_lstm_cell/bias*
T0*
_output_shapes
:P
[
rnn/basic_lstm_cell/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
a
rnn/basic_lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Й
rnn/basic_lstm_cell/concatConcatV2split"rnn/BasicLSTMCellZeroState/zeros_1rnn/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Й
rnn/basic_lstm_cell/MatMulMatMulrnn/basic_lstm_cell/concatrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџP
Њ
rnn/basic_lstm_cell/BiasAddBiasAddrnn/basic_lstm_cell/MatMulrnn/basic_lstm_cell/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџP
]
rnn/basic_lstm_cell/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
ж
rnn/basic_lstm_cell/splitSplitrnn/basic_lstm_cell/Constrnn/basic_lstm_cell/BiasAdd*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
`
rnn/basic_lstm_cell/Const_2Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/AddAddrnn/basic_lstm_cell/split:2rnn/basic_lstm_cell/Const_2*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/SigmoidSigmoidrnn/basic_lstm_cell/Add*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/MulMul rnn/BasicLSTMCellZeroState/zerosrnn/basic_lstm_cell/Sigmoid*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Sigmoid_1Sigmoidrnn/basic_lstm_cell/split*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/TanhTanhrnn/basic_lstm_cell/split:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_1rnn/basic_lstm_cell/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_1Addrnn/basic_lstm_cell/Mulrnn/basic_lstm_cell/Mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/Tanh_1Tanhrnn/basic_lstm_cell/Add_1*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_2Sigmoidrnn/basic_lstm_cell/split:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_2Mulrnn/basic_lstm_cell/Tanh_1rnn/basic_lstm_cell/Sigmoid_2*
T0*'
_output_shapes
:џџџџџџџџџ
]
rnn/basic_lstm_cell/Const_3Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
rnn/basic_lstm_cell/concat_1ConcatV2split:1rnn/basic_lstm_cell/Mul_2!rnn/basic_lstm_cell/concat_1/axis*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_1MatMulrnn/basic_lstm_cell/concat_1rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_1BiasAddrnn/basic_lstm_cell/MatMul_1rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
]
rnn/basic_lstm_cell/Const_4Const*
value	B :*
dtype0*
_output_shapes
: 
м
rnn/basic_lstm_cell/split_1Splitrnn/basic_lstm_cell/Const_3rnn/basic_lstm_cell/BiasAdd_1*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
`
rnn/basic_lstm_cell/Const_5Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 

rnn/basic_lstm_cell/Add_2Addrnn/basic_lstm_cell/split_1:2rnn/basic_lstm_cell/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Sigmoid_3Sigmoidrnn/basic_lstm_cell/Add_2*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_3Mulrnn/basic_lstm_cell/Add_1rnn/basic_lstm_cell/Sigmoid_3*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_4Sigmoidrnn/basic_lstm_cell/split_1*
T0*'
_output_shapes
:џџџџџџџџџ
s
rnn/basic_lstm_cell/Tanh_2Tanhrnn/basic_lstm_cell/split_1:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_4Mulrnn/basic_lstm_cell/Sigmoid_4rnn/basic_lstm_cell/Tanh_2*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_3Addrnn/basic_lstm_cell/Mul_3rnn/basic_lstm_cell/Mul_4*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/Tanh_3Tanhrnn/basic_lstm_cell/Add_3*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_5Sigmoidrnn/basic_lstm_cell/split_1:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_5Mulrnn/basic_lstm_cell/Tanh_3rnn/basic_lstm_cell/Sigmoid_5*
T0*'
_output_shapes
:џџџџџџџџџ
]
rnn/basic_lstm_cell/Const_6Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_2/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
rnn/basic_lstm_cell/concat_2ConcatV2split:2rnn/basic_lstm_cell/Mul_5!rnn/basic_lstm_cell/concat_2/axis*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_2MatMulrnn/basic_lstm_cell/concat_2rnn/basic_lstm_cell/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_2BiasAddrnn/basic_lstm_cell/MatMul_2rnn/basic_lstm_cell/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџP
]
rnn/basic_lstm_cell/Const_7Const*
value	B :*
dtype0*
_output_shapes
: 
м
rnn/basic_lstm_cell/split_2Splitrnn/basic_lstm_cell/Const_6rnn/basic_lstm_cell/BiasAdd_2*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
`
rnn/basic_lstm_cell/Const_8Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_4Addrnn/basic_lstm_cell/split_2:2rnn/basic_lstm_cell/Const_8*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Sigmoid_6Sigmoidrnn/basic_lstm_cell/Add_4*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_6Mulrnn/basic_lstm_cell/Add_3rnn/basic_lstm_cell/Sigmoid_6*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_7Sigmoidrnn/basic_lstm_cell/split_2*
T0*'
_output_shapes
:џџџџџџџџџ
s
rnn/basic_lstm_cell/Tanh_4Tanhrnn/basic_lstm_cell/split_2:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_7Mulrnn/basic_lstm_cell/Sigmoid_7rnn/basic_lstm_cell/Tanh_4*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_5Addrnn/basic_lstm_cell/Mul_6rnn/basic_lstm_cell/Mul_7*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/Tanh_5Tanhrnn/basic_lstm_cell/Add_5*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_8Sigmoidrnn/basic_lstm_cell/split_2:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_8Mulrnn/basic_lstm_cell/Tanh_5rnn/basic_lstm_cell/Sigmoid_8*
T0*'
_output_shapes
:џџџџџџџџџ
]
rnn/basic_lstm_cell/Const_9Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_3/axisConst*
dtype0*
value	B :*
_output_shapes
: 
Ж
rnn/basic_lstm_cell/concat_3ConcatV2split:3rnn/basic_lstm_cell/Mul_8!rnn/basic_lstm_cell/concat_3/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_3MatMulrnn/basic_lstm_cell/concat_3rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_3BiasAddrnn/basic_lstm_cell/MatMul_3rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_10Const*
value	B :*
dtype0*
_output_shapes
: 
м
rnn/basic_lstm_cell/split_3Splitrnn/basic_lstm_cell/Const_9rnn/basic_lstm_cell/BiasAdd_3*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_11Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_6Addrnn/basic_lstm_cell/split_3:2rnn/basic_lstm_cell/Const_11*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Sigmoid_9Sigmoidrnn/basic_lstm_cell/Add_6*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_9Mulrnn/basic_lstm_cell/Add_5rnn/basic_lstm_cell/Sigmoid_9*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_10Sigmoidrnn/basic_lstm_cell/split_3*
T0*'
_output_shapes
:џџџџџџџџџ
s
rnn/basic_lstm_cell/Tanh_6Tanhrnn/basic_lstm_cell/split_3:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_10Mulrnn/basic_lstm_cell/Sigmoid_10rnn/basic_lstm_cell/Tanh_6*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_7Addrnn/basic_lstm_cell/Mul_9rnn/basic_lstm_cell/Mul_10*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/Tanh_7Tanhrnn/basic_lstm_cell/Add_7*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_11Sigmoidrnn/basic_lstm_cell/split_3:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_11Mulrnn/basic_lstm_cell/Tanh_7rnn/basic_lstm_cell/Sigmoid_11*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_12Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_4/axisConst*
value	B :*
dtype0*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_4ConcatV2split:4rnn/basic_lstm_cell/Mul_11!rnn/basic_lstm_cell/concat_4/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_4MatMulrnn/basic_lstm_cell/concat_4rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_4BiasAddrnn/basic_lstm_cell/MatMul_4rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_13Const*
dtype0*
value	B :*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_4Splitrnn/basic_lstm_cell/Const_12rnn/basic_lstm_cell/BiasAdd_4*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_14Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_8Addrnn/basic_lstm_cell/split_4:2rnn/basic_lstm_cell/Const_14*
T0*'
_output_shapes
:џџџџџџџџџ
v
rnn/basic_lstm_cell/Sigmoid_12Sigmoidrnn/basic_lstm_cell/Add_8*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_12Mulrnn/basic_lstm_cell/Add_7rnn/basic_lstm_cell/Sigmoid_12*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_13Sigmoidrnn/basic_lstm_cell/split_4*
T0*'
_output_shapes
:џџџџџџџџџ
s
rnn/basic_lstm_cell/Tanh_8Tanhrnn/basic_lstm_cell/split_4:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_13Mulrnn/basic_lstm_cell/Sigmoid_13rnn/basic_lstm_cell/Tanh_8*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_9Addrnn/basic_lstm_cell/Mul_12rnn/basic_lstm_cell/Mul_13*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/Tanh_9Tanhrnn/basic_lstm_cell/Add_9*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_14Sigmoidrnn/basic_lstm_cell/split_4:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_14Mulrnn/basic_lstm_cell/Tanh_9rnn/basic_lstm_cell/Sigmoid_14*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_15Const*
dtype0*
value	B :*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_5/axisConst*
value	B :*
dtype0*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_5ConcatV2split:5rnn/basic_lstm_cell/Mul_14!rnn/basic_lstm_cell/concat_5/axis*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_5MatMulrnn/basic_lstm_cell/concat_5rnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_5BiasAddrnn/basic_lstm_cell/MatMul_5rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_16Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_5Splitrnn/basic_lstm_cell/Const_15rnn/basic_lstm_cell/BiasAdd_5*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_17Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_10Addrnn/basic_lstm_cell/split_5:2rnn/basic_lstm_cell/Const_17*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_15Sigmoidrnn/basic_lstm_cell/Add_10*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_15Mulrnn/basic_lstm_cell/Add_9rnn/basic_lstm_cell/Sigmoid_15*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_16Sigmoidrnn/basic_lstm_cell/split_5*
T0*'
_output_shapes
:џџџџџџџџџ
t
rnn/basic_lstm_cell/Tanh_10Tanhrnn/basic_lstm_cell/split_5:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_16Mulrnn/basic_lstm_cell/Sigmoid_16rnn/basic_lstm_cell/Tanh_10*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_11Addrnn/basic_lstm_cell/Mul_15rnn/basic_lstm_cell/Mul_16*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_11Tanhrnn/basic_lstm_cell/Add_11*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_17Sigmoidrnn/basic_lstm_cell/split_5:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_17Mulrnn/basic_lstm_cell/Tanh_11rnn/basic_lstm_cell/Sigmoid_17*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_18Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_6/axisConst*
value	B :*
dtype0*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_6ConcatV2split:6rnn/basic_lstm_cell/Mul_17!rnn/basic_lstm_cell/concat_6/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_6MatMulrnn/basic_lstm_cell/concat_6rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_6BiasAddrnn/basic_lstm_cell/MatMul_6rnn/basic_lstm_cell/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_19Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_6Splitrnn/basic_lstm_cell/Const_18rnn/basic_lstm_cell/BiasAdd_6*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_20Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_12Addrnn/basic_lstm_cell/split_6:2rnn/basic_lstm_cell/Const_20*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_18Sigmoidrnn/basic_lstm_cell/Add_12*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_18Mulrnn/basic_lstm_cell/Add_11rnn/basic_lstm_cell/Sigmoid_18*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_19Sigmoidrnn/basic_lstm_cell/split_6*
T0*'
_output_shapes
:џџџџџџџџџ
t
rnn/basic_lstm_cell/Tanh_12Tanhrnn/basic_lstm_cell/split_6:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_19Mulrnn/basic_lstm_cell/Sigmoid_19rnn/basic_lstm_cell/Tanh_12*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_13Addrnn/basic_lstm_cell/Mul_18rnn/basic_lstm_cell/Mul_19*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_13Tanhrnn/basic_lstm_cell/Add_13*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_20Sigmoidrnn/basic_lstm_cell/split_6:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_20Mulrnn/basic_lstm_cell/Tanh_13rnn/basic_lstm_cell/Sigmoid_20*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_21Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_7/axisConst*
dtype0*
value	B :*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_7ConcatV2split:7rnn/basic_lstm_cell/Mul_20!rnn/basic_lstm_cell/concat_7/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_7MatMulrnn/basic_lstm_cell/concat_7rnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_7BiasAddrnn/basic_lstm_cell/MatMul_7rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_22Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_7Splitrnn/basic_lstm_cell/Const_21rnn/basic_lstm_cell/BiasAdd_7*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_23Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_14Addrnn/basic_lstm_cell/split_7:2rnn/basic_lstm_cell/Const_23*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_21Sigmoidrnn/basic_lstm_cell/Add_14*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_21Mulrnn/basic_lstm_cell/Add_13rnn/basic_lstm_cell/Sigmoid_21*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_22Sigmoidrnn/basic_lstm_cell/split_7*
T0*'
_output_shapes
:џџџџџџџџџ
t
rnn/basic_lstm_cell/Tanh_14Tanhrnn/basic_lstm_cell/split_7:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_22Mulrnn/basic_lstm_cell/Sigmoid_22rnn/basic_lstm_cell/Tanh_14*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_15Addrnn/basic_lstm_cell/Mul_21rnn/basic_lstm_cell/Mul_22*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_15Tanhrnn/basic_lstm_cell/Add_15*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_23Sigmoidrnn/basic_lstm_cell/split_7:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_23Mulrnn/basic_lstm_cell/Tanh_15rnn/basic_lstm_cell/Sigmoid_23*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_24Const*
dtype0*
value	B :*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_8/axisConst*
value	B :*
dtype0*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_8ConcatV2split:8rnn/basic_lstm_cell/Mul_23!rnn/basic_lstm_cell/concat_8/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_8MatMulrnn/basic_lstm_cell/concat_8rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_8BiasAddrnn/basic_lstm_cell/MatMul_8rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_25Const*
dtype0*
value	B :*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_8Splitrnn/basic_lstm_cell/Const_24rnn/basic_lstm_cell/BiasAdd_8*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_26Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_16Addrnn/basic_lstm_cell/split_8:2rnn/basic_lstm_cell/Const_26*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_24Sigmoidrnn/basic_lstm_cell/Add_16*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_24Mulrnn/basic_lstm_cell/Add_15rnn/basic_lstm_cell/Sigmoid_24*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_25Sigmoidrnn/basic_lstm_cell/split_8*
T0*'
_output_shapes
:џџџџџџџџџ
t
rnn/basic_lstm_cell/Tanh_16Tanhrnn/basic_lstm_cell/split_8:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_25Mulrnn/basic_lstm_cell/Sigmoid_25rnn/basic_lstm_cell/Tanh_16*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_17Addrnn/basic_lstm_cell/Mul_24rnn/basic_lstm_cell/Mul_25*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_17Tanhrnn/basic_lstm_cell/Add_17*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_26Sigmoidrnn/basic_lstm_cell/split_8:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_26Mulrnn/basic_lstm_cell/Tanh_17rnn/basic_lstm_cell/Sigmoid_26*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_27Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_9/axisConst*
value	B :*
dtype0*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_9ConcatV2split:9rnn/basic_lstm_cell/Mul_26!rnn/basic_lstm_cell/concat_9/axis*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_9MatMulrnn/basic_lstm_cell/concat_9rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_9BiasAddrnn/basic_lstm_cell/MatMul_9rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_28Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_9Splitrnn/basic_lstm_cell/Const_27rnn/basic_lstm_cell/BiasAdd_9*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_29Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_18Addrnn/basic_lstm_cell/split_9:2rnn/basic_lstm_cell/Const_29*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_27Sigmoidrnn/basic_lstm_cell/Add_18*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_27Mulrnn/basic_lstm_cell/Add_17rnn/basic_lstm_cell/Sigmoid_27*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_28Sigmoidrnn/basic_lstm_cell/split_9*
T0*'
_output_shapes
:џџџџџџџџџ
t
rnn/basic_lstm_cell/Tanh_18Tanhrnn/basic_lstm_cell/split_9:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_28Mulrnn/basic_lstm_cell/Sigmoid_28rnn/basic_lstm_cell/Tanh_18*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_19Addrnn/basic_lstm_cell/Mul_27rnn/basic_lstm_cell/Mul_28*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_19Tanhrnn/basic_lstm_cell/Add_19*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_29Sigmoidrnn/basic_lstm_cell/split_9:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_29Mulrnn/basic_lstm_cell/Tanh_19rnn/basic_lstm_cell/Sigmoid_29*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_30Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_10/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_10ConcatV2split:10rnn/basic_lstm_cell/Mul_29"rnn/basic_lstm_cell/concat_10/axis*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_10MatMulrnn/basic_lstm_cell/concat_10rnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_10BiasAddrnn/basic_lstm_cell/MatMul_10rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_31Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_10Splitrnn/basic_lstm_cell/Const_30rnn/basic_lstm_cell/BiasAdd_10*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_32Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_20Addrnn/basic_lstm_cell/split_10:2rnn/basic_lstm_cell/Const_32*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_30Sigmoidrnn/basic_lstm_cell/Add_20*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_30Mulrnn/basic_lstm_cell/Add_19rnn/basic_lstm_cell/Sigmoid_30*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_31Sigmoidrnn/basic_lstm_cell/split_10*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_20Tanhrnn/basic_lstm_cell/split_10:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_31Mulrnn/basic_lstm_cell/Sigmoid_31rnn/basic_lstm_cell/Tanh_20*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_21Addrnn/basic_lstm_cell/Mul_30rnn/basic_lstm_cell/Mul_31*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_21Tanhrnn/basic_lstm_cell/Add_21*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_32Sigmoidrnn/basic_lstm_cell/split_10:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_32Mulrnn/basic_lstm_cell/Tanh_21rnn/basic_lstm_cell/Sigmoid_32*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_33Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_11/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_11ConcatV2split:11rnn/basic_lstm_cell/Mul_32"rnn/basic_lstm_cell/concat_11/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_11MatMulrnn/basic_lstm_cell/concat_11rnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_11BiasAddrnn/basic_lstm_cell/MatMul_11rnn/basic_lstm_cell/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_34Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_11Splitrnn/basic_lstm_cell/Const_33rnn/basic_lstm_cell/BiasAdd_11*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_35Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_22Addrnn/basic_lstm_cell/split_11:2rnn/basic_lstm_cell/Const_35*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_33Sigmoidrnn/basic_lstm_cell/Add_22*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_33Mulrnn/basic_lstm_cell/Add_21rnn/basic_lstm_cell/Sigmoid_33*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_34Sigmoidrnn/basic_lstm_cell/split_11*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_22Tanhrnn/basic_lstm_cell/split_11:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_34Mulrnn/basic_lstm_cell/Sigmoid_34rnn/basic_lstm_cell/Tanh_22*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_23Addrnn/basic_lstm_cell/Mul_33rnn/basic_lstm_cell/Mul_34*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_23Tanhrnn/basic_lstm_cell/Add_23*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_35Sigmoidrnn/basic_lstm_cell/split_11:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_35Mulrnn/basic_lstm_cell/Tanh_23rnn/basic_lstm_cell/Sigmoid_35*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_36Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_12/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_12ConcatV2split:12rnn/basic_lstm_cell/Mul_35"rnn/basic_lstm_cell/concat_12/axis*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_12MatMulrnn/basic_lstm_cell/concat_12rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_12BiasAddrnn/basic_lstm_cell/MatMul_12rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_37Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_12Splitrnn/basic_lstm_cell/Const_36rnn/basic_lstm_cell/BiasAdd_12*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_38Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_24Addrnn/basic_lstm_cell/split_12:2rnn/basic_lstm_cell/Const_38*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_36Sigmoidrnn/basic_lstm_cell/Add_24*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_36Mulrnn/basic_lstm_cell/Add_23rnn/basic_lstm_cell/Sigmoid_36*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_37Sigmoidrnn/basic_lstm_cell/split_12*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_24Tanhrnn/basic_lstm_cell/split_12:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_37Mulrnn/basic_lstm_cell/Sigmoid_37rnn/basic_lstm_cell/Tanh_24*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_25Addrnn/basic_lstm_cell/Mul_36rnn/basic_lstm_cell/Mul_37*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_25Tanhrnn/basic_lstm_cell/Add_25*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_38Sigmoidrnn/basic_lstm_cell/split_12:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_38Mulrnn/basic_lstm_cell/Tanh_25rnn/basic_lstm_cell/Sigmoid_38*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_39Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_13/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_13ConcatV2split:13rnn/basic_lstm_cell/Mul_38"rnn/basic_lstm_cell/concat_13/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_13MatMulrnn/basic_lstm_cell/concat_13rnn/basic_lstm_cell/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_13BiasAddrnn/basic_lstm_cell/MatMul_13rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_40Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_13Splitrnn/basic_lstm_cell/Const_39rnn/basic_lstm_cell/BiasAdd_13*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_41Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_26Addrnn/basic_lstm_cell/split_13:2rnn/basic_lstm_cell/Const_41*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_39Sigmoidrnn/basic_lstm_cell/Add_26*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_39Mulrnn/basic_lstm_cell/Add_25rnn/basic_lstm_cell/Sigmoid_39*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_40Sigmoidrnn/basic_lstm_cell/split_13*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_26Tanhrnn/basic_lstm_cell/split_13:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_40Mulrnn/basic_lstm_cell/Sigmoid_40rnn/basic_lstm_cell/Tanh_26*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_27Addrnn/basic_lstm_cell/Mul_39rnn/basic_lstm_cell/Mul_40*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_27Tanhrnn/basic_lstm_cell/Add_27*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_41Sigmoidrnn/basic_lstm_cell/split_13:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_41Mulrnn/basic_lstm_cell/Tanh_27rnn/basic_lstm_cell/Sigmoid_41*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_42Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_14/axisConst*
dtype0*
value	B :*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_14ConcatV2split:14rnn/basic_lstm_cell/Mul_41"rnn/basic_lstm_cell/concat_14/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_14MatMulrnn/basic_lstm_cell/concat_14rnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_14BiasAddrnn/basic_lstm_cell/MatMul_14rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_43Const*
dtype0*
value	B :*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_14Splitrnn/basic_lstm_cell/Const_42rnn/basic_lstm_cell/BiasAdd_14*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_44Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_28Addrnn/basic_lstm_cell/split_14:2rnn/basic_lstm_cell/Const_44*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_42Sigmoidrnn/basic_lstm_cell/Add_28*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_42Mulrnn/basic_lstm_cell/Add_27rnn/basic_lstm_cell/Sigmoid_42*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_43Sigmoidrnn/basic_lstm_cell/split_14*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_28Tanhrnn/basic_lstm_cell/split_14:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_43Mulrnn/basic_lstm_cell/Sigmoid_43rnn/basic_lstm_cell/Tanh_28*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_29Addrnn/basic_lstm_cell/Mul_42rnn/basic_lstm_cell/Mul_43*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_29Tanhrnn/basic_lstm_cell/Add_29*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_44Sigmoidrnn/basic_lstm_cell/split_14:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_44Mulrnn/basic_lstm_cell/Tanh_29rnn/basic_lstm_cell/Sigmoid_44*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_45Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_15/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_15ConcatV2split:15rnn/basic_lstm_cell/Mul_44"rnn/basic_lstm_cell/concat_15/axis*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_15MatMulrnn/basic_lstm_cell/concat_15rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_15BiasAddrnn/basic_lstm_cell/MatMul_15rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_46Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_15Splitrnn/basic_lstm_cell/Const_45rnn/basic_lstm_cell/BiasAdd_15*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_47Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_30Addrnn/basic_lstm_cell/split_15:2rnn/basic_lstm_cell/Const_47*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_45Sigmoidrnn/basic_lstm_cell/Add_30*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_45Mulrnn/basic_lstm_cell/Add_29rnn/basic_lstm_cell/Sigmoid_45*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_46Sigmoidrnn/basic_lstm_cell/split_15*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_30Tanhrnn/basic_lstm_cell/split_15:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_46Mulrnn/basic_lstm_cell/Sigmoid_46rnn/basic_lstm_cell/Tanh_30*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_31Addrnn/basic_lstm_cell/Mul_45rnn/basic_lstm_cell/Mul_46*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_31Tanhrnn/basic_lstm_cell/Add_31*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_47Sigmoidrnn/basic_lstm_cell/split_15:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_47Mulrnn/basic_lstm_cell/Tanh_31rnn/basic_lstm_cell/Sigmoid_47*
T0*'
_output_shapes
:џџџџџџџџџ

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *   П*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *   ?*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
щ
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
seed2м*
seedвЎК	*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
Ю
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
р
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
в
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
Ё
dense/kernel
VariableV2*
shape
:*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container *
_output_shapes

:
Ч
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
u
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes

:

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:


dense/bias
VariableV2*
_class
loc:@dense/bias*
dtype0*
	container *
shape:*
shared_name *
_output_shapes
:
В
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
k
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes
:

dense/MatMulMatMulrnn/basic_lstm_cell/Mul_47dense/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ

dense/BiasAddBiasAdddense/MatMuldense/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ

$mean_squared_error/SquaredDifferenceSquaredDifferencedense/BiasAddIteratorGetNext:1*
T0*'
_output_shapes
:џџџџџџџџџ
t
/mean_squared_error/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
x
5mean_squared_error/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
v
4mean_squared_error/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 

4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
u
3mean_squared_error/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
K
Cmean_squared_error/assert_broadcastable/static_scalar_check_successNoOp
Љ
mean_squared_error/ToFloat_3/xConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/ToFloat_3/x*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
mean_squared_error/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB"       *
_output_shapes
:

mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Б
&mean_squared_error/num_present/Equal/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

$mean_squared_error/num_present/EqualEqualmean_squared_error/ToFloat_3/x&mean_squared_error/num_present/Equal/y*
T0*
_output_shapes
: 
Д
)mean_squared_error/num_present/zeros_likeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
З
.mean_squared_error/num_present/ones_like/ShapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Й
.mean_squared_error/num_present/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
У
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
Ы
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
м
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
к
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
ќ
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
й
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
Џ
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpD^mean_squared_error/assert_broadcastable/static_scalar_check_success
Ю
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
Џ
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
Ь
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:џџџџџџџџџ
Л
$mean_squared_error/num_present/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
Л
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ѓ
mean_squared_error/Const_1ConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 

mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ї
mean_squared_error/Greater/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

mean_squared_error/GreaterGreatermean_squared_error/num_presentmean_squared_error/Greater/y*
T0*
_output_shapes
: 
Ѕ
mean_squared_error/Equal/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
~
mean_squared_error/EqualEqualmean_squared_error/num_presentmean_squared_error/Equal/y*
T0*
_output_shapes
: 
Ћ
"mean_squared_error/ones_like/ShapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
­
"mean_squared_error/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

mean_squared_error/ones_likeFill"mean_squared_error/ones_like/Shape"mean_squared_error/ones_like/Const*
T0*

index_type0*
_output_shapes
: 

mean_squared_error/SelectSelectmean_squared_error/Equalmean_squared_error/ones_likemean_squared_error/num_present*
T0*
_output_shapes
: 
w
mean_squared_error/divRealDivmean_squared_error/Sum_1mean_squared_error/Select*
T0*
_output_shapes
: 
Ј
mean_squared_error/zeros_likeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

mean_squared_error/valueSelectmean_squared_error/Greatermean_squared_error/divmean_squared_error/zeros_like*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
w
2gradients/mean_squared_error/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Й
.gradients/mean_squared_error/value_grad/SelectSelectmean_squared_error/Greatergradients/Fill2gradients/mean_squared_error/value_grad/zeros_like*
T0*
_output_shapes
: 
Л
0gradients/mean_squared_error/value_grad/Select_1Selectmean_squared_error/Greater2gradients/mean_squared_error/value_grad/zeros_likegradients/Fill*
T0*
_output_shapes
: 
Є
8gradients/mean_squared_error/value_grad/tuple/group_depsNoOp/^gradients/mean_squared_error/value_grad/Select1^gradients/mean_squared_error/value_grad/Select_1

@gradients/mean_squared_error/value_grad/tuple/control_dependencyIdentity.gradients/mean_squared_error/value_grad/Select9^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/mean_squared_error/value_grad/Select*
_output_shapes
: 
Ё
Bgradients/mean_squared_error/value_grad/tuple/control_dependency_1Identity0gradients/mean_squared_error/value_grad/Select_19^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/mean_squared_error/value_grad/Select_1*
_output_shapes
: 
n
+gradients/mean_squared_error/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
p
-gradients/mean_squared_error/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
э
;gradients/mean_squared_error/div_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/div_grad/Shape-gradients/mean_squared_error/div_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ж
-gradients/mean_squared_error/div_grad/RealDivRealDiv@gradients/mean_squared_error/value_grad/tuple/control_dependencymean_squared_error/Select*
T0*
_output_shapes
: 
м
)gradients/mean_squared_error/div_grad/SumSum-gradients/mean_squared_error/div_grad/RealDiv;gradients/mean_squared_error/div_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
П
-gradients/mean_squared_error/div_grad/ReshapeReshape)gradients/mean_squared_error/div_grad/Sum+gradients/mean_squared_error/div_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
k
)gradients/mean_squared_error/div_grad/NegNegmean_squared_error/Sum_1*
T0*
_output_shapes
: 
Ё
/gradients/mean_squared_error/div_grad/RealDiv_1RealDiv)gradients/mean_squared_error/div_grad/Negmean_squared_error/Select*
T0*
_output_shapes
: 
Ї
/gradients/mean_squared_error/div_grad/RealDiv_2RealDiv/gradients/mean_squared_error/div_grad/RealDiv_1mean_squared_error/Select*
T0*
_output_shapes
: 
Ф
)gradients/mean_squared_error/div_grad/mulMul@gradients/mean_squared_error/value_grad/tuple/control_dependency/gradients/mean_squared_error/div_grad/RealDiv_2*
T0*
_output_shapes
: 
м
+gradients/mean_squared_error/div_grad/Sum_1Sum)gradients/mean_squared_error/div_grad/mul=gradients/mean_squared_error/div_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Х
/gradients/mean_squared_error/div_grad/Reshape_1Reshape+gradients/mean_squared_error/div_grad/Sum_1-gradients/mean_squared_error/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
 
6gradients/mean_squared_error/div_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/div_grad/Reshape0^gradients/mean_squared_error/div_grad/Reshape_1

>gradients/mean_squared_error/div_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/div_grad/Reshape7^gradients/mean_squared_error/div_grad/tuple/group_deps*@
_class6
42loc:@gradients/mean_squared_error/div_grad/Reshape*
T0*
_output_shapes
: 

@gradients/mean_squared_error/div_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/div_grad/Reshape_17^gradients/mean_squared_error/div_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/div_grad/Reshape_1*
_output_shapes
: 
x
5gradients/mean_squared_error/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
р
/gradients/mean_squared_error/Sum_1_grad/ReshapeReshape>gradients/mean_squared_error/div_grad/tuple/control_dependency5gradients/mean_squared_error/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
p
-gradients/mean_squared_error/Sum_1_grad/ConstConst*
dtype0*
valueB *
_output_shapes
: 
Ч
,gradients/mean_squared_error/Sum_1_grad/TileTile/gradients/mean_squared_error/Sum_1_grad/Reshape-gradients/mean_squared_error/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
x
3gradients/mean_squared_error/Select_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ы
/gradients/mean_squared_error/Select_grad/SelectSelectmean_squared_error/Equal@gradients/mean_squared_error/div_grad/tuple/control_dependency_13gradients/mean_squared_error/Select_grad/zeros_like*
T0*
_output_shapes
: 
э
1gradients/mean_squared_error/Select_grad/Select_1Selectmean_squared_error/Equal3gradients/mean_squared_error/Select_grad/zeros_like@gradients/mean_squared_error/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
Ї
9gradients/mean_squared_error/Select_grad/tuple/group_depsNoOp0^gradients/mean_squared_error/Select_grad/Select2^gradients/mean_squared_error/Select_grad/Select_1

Agradients/mean_squared_error/Select_grad/tuple/control_dependencyIdentity/gradients/mean_squared_error/Select_grad/Select:^gradients/mean_squared_error/Select_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/Select_grad/Select*
_output_shapes
: 
Ѕ
Cgradients/mean_squared_error/Select_grad/tuple/control_dependency_1Identity1gradients/mean_squared_error/Select_grad/Select_1:^gradients/mean_squared_error/Select_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/mean_squared_error/Select_grad/Select_1*
_output_shapes
: 

3gradients/mean_squared_error/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
в
-gradients/mean_squared_error/Sum_grad/ReshapeReshape,gradients/mean_squared_error/Sum_1_grad/Tile3gradients/mean_squared_error/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:

+gradients/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
T0*
out_type0*
_output_shapes
:
в
*gradients/mean_squared_error/Sum_grad/TileTile-gradients/mean_squared_error/Sum_grad/Reshape+gradients/mean_squared_error/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ

+gradients/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
p
-gradients/mean_squared_error/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
э
;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/Mul_grad/Shape-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ў
)gradients/mean_squared_error/Mul_grad/MulMul*gradients/mean_squared_error/Sum_grad/Tilemean_squared_error/ToFloat_3/x*
T0*'
_output_shapes
:џџџџџџџџџ
и
)gradients/mean_squared_error/Mul_grad/SumSum)gradients/mean_squared_error/Mul_grad/Mul;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
а
-gradients/mean_squared_error/Mul_grad/ReshapeReshape)gradients/mean_squared_error/Mul_grad/Sum+gradients/mean_squared_error/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ж
+gradients/mean_squared_error/Mul_grad/Mul_1Mul$mean_squared_error/SquaredDifference*gradients/mean_squared_error/Sum_grad/Tile*
T0*'
_output_shapes
:џџџџџџџџџ
о
+gradients/mean_squared_error/Mul_grad/Sum_1Sum+gradients/mean_squared_error/Mul_grad/Mul_1=gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Х
/gradients/mean_squared_error/Mul_grad/Reshape_1Reshape+gradients/mean_squared_error/Mul_grad/Sum_1-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
 
6gradients/mean_squared_error/Mul_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/Mul_grad/Reshape0^gradients/mean_squared_error/Mul_grad/Reshape_1
І
>gradients/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/Mul_grad/Reshape7^gradients/mean_squared_error/Mul_grad/tuple/group_deps*@
_class6
42loc:@gradients/mean_squared_error/Mul_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ

@gradients/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/Mul_grad/Reshape_17^gradients/mean_squared_error/Mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/Mul_grad/Reshape_1*
_output_shapes
: 

;gradients/mean_squared_error/num_present_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
љ
5gradients/mean_squared_error/num_present_grad/ReshapeReshapeCgradients/mean_squared_error/Select_grad/tuple/control_dependency_1;gradients/mean_squared_error/num_present_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
Ѓ
3gradients/mean_squared_error/num_present_grad/ShapeShape0mean_squared_error/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
ъ
2gradients/mean_squared_error/num_present_grad/TileTile5gradients/mean_squared_error/num_present_grad/Reshape3gradients/mean_squared_error/num_present_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ

Egradients/mean_squared_error/num_present/broadcast_weights_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
С
Ggradients/mean_squared_error/num_present/broadcast_weights_grad/Shape_1Shape:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
out_type0*
_output_shapes
:
Л
Ugradients/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/mean_squared_error/num_present/broadcast_weights_grad/ShapeGgradients/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ь
Cgradients/mean_squared_error/num_present/broadcast_weights_grad/MulMul2gradients/mean_squared_error/num_present_grad/Tile:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:џџџџџџџџџ
І
Cgradients/mean_squared_error/num_present/broadcast_weights_grad/SumSumCgradients/mean_squared_error/num_present/broadcast_weights_grad/MulUgradients/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:

Ggradients/mean_squared_error/num_present/broadcast_weights_grad/ReshapeReshapeCgradients/mean_squared_error/num_present/broadcast_weights_grad/SumEgradients/mean_squared_error/num_present/broadcast_weights_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
й
Egradients/mean_squared_error/num_present/broadcast_weights_grad/Mul_1Mul%mean_squared_error/num_present/Select2gradients/mean_squared_error/num_present_grad/Tile*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
Egradients/mean_squared_error/num_present/broadcast_weights_grad/Sum_1SumEgradients/mean_squared_error/num_present/broadcast_weights_grad/Mul_1Wgradients/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Є
Igradients/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1ReshapeEgradients/mean_squared_error/num_present/broadcast_weights_grad/Sum_1Ggradients/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ю
Pgradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_depsNoOpH^gradients/mean_squared_error/num_present/broadcast_weights_grad/ReshapeJ^gradients/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1
§
Xgradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityGgradients/mean_squared_error/num_present/broadcast_weights_grad/ReshapeQ^gradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/mean_squared_error/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: 

Zgradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityIgradients/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1Q^gradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
 
Ogradients/mean_squared_error/num_present/broadcast_weights/ones_like_grad/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
П
Mgradients/mean_squared_error/num_present/broadcast_weights/ones_like_grad/SumSumZgradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1Ogradients/mean_squared_error/num_present/broadcast_weights/ones_like_grad/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 

9gradients/mean_squared_error/SquaredDifference_grad/ShapeShapedense/BiasAdd*
T0*
out_type0*
_output_shapes
:

;gradients/mean_squared_error/SquaredDifference_grad/Shape_1ShapeIteratorGetNext:1*
out_type0*
T0*
_output_shapes
:

Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/mean_squared_error/SquaredDifference_grad/Shape;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Р
:gradients/mean_squared_error/SquaredDifference_grad/scalarConst?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
valueB
 *   @*
dtype0*
_output_shapes
: 
ь
7gradients/mean_squared_error/SquaredDifference_grad/mulMul:gradients/mean_squared_error/SquaredDifference_grad/scalar>gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
г
7gradients/mean_squared_error/SquaredDifference_grad/subSubdense/BiasAddIteratorGetNext:1?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ф
9gradients/mean_squared_error/SquaredDifference_grad/mul_1Mul7gradients/mean_squared_error/SquaredDifference_grad/mul7gradients/mean_squared_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ

7gradients/mean_squared_error/SquaredDifference_grad/SumSum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
њ
;gradients/mean_squared_error/SquaredDifference_grad/ReshapeReshape7gradients/mean_squared_error/SquaredDifference_grad/Sum9gradients/mean_squared_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

9gradients/mean_squared_error/SquaredDifference_grad/Sum_1Sum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Kgradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:

=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape9gradients/mean_squared_error/SquaredDifference_grad/Sum_1;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Џ
7gradients/mean_squared_error/SquaredDifference_grad/NegNeg=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
Dgradients/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp8^gradients/mean_squared_error/SquaredDifference_grad/Neg<^gradients/mean_squared_error/SquaredDifference_grad/Reshape
о
Lgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity;gradients/mean_squared_error/SquaredDifference_grad/ReshapeE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
и
Ngradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity7gradients/mean_squared_error/SquaredDifference_grad/NegE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/mean_squared_error/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
С
(gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGradLgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:
Џ
-gradients/dense/BiasAdd_grad/tuple/group_depsNoOp)^gradients/dense/BiasAdd_grad/BiasAddGradM^gradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency
С
5gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentityLgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency.^gradients/dense/BiasAdd_grad/tuple/group_deps*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
џ
7gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/dense/BiasAdd_grad/BiasAddGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ю
"gradients/dense/MatMul_grad/MatMulMatMul5gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
а
$gradients/dense/MatMul_grad/MatMul_1MatMulrnn/basic_lstm_cell/Mul_475gradients/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:

,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
ќ
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
љ
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes

:

/gradients/rnn/basic_lstm_cell/Mul_47_grad/ShapeShapernn/basic_lstm_cell/Tanh_31*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_47_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_47*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_47_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_47_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_47_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
М
-gradients/rnn/basic_lstm_cell/Mul_47_grad/MulMul4gradients/dense/MatMul_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_47*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_47_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_47_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_47_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_47_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_47_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_47_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Л
/gradients/rnn/basic_lstm_cell/Mul_47_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_314gradients/dense/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_47_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_47_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_47_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_47_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_47_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_31_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_31Bgradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_47_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_47Dgradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_31_grad/ShapeShapernn/basic_lstm_cell/Mul_45*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_31_grad/Shape_1Shapernn/basic_lstm_cell/Mul_46*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_31_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_31_grad/Shape1gradients/rnn/basic_lstm_cell/Add_31_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ъ
-gradients/rnn/basic_lstm_cell/Add_31_grad/SumSum3gradients/rnn/basic_lstm_cell/Tanh_31_grad/TanhGrad?gradients/rnn/basic_lstm_cell/Add_31_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_31_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_31_grad/Sum/gradients/rnn/basic_lstm_cell/Add_31_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ю
/gradients/rnn/basic_lstm_cell/Add_31_grad/Sum_1Sum3gradients/rnn/basic_lstm_cell/Tanh_31_grad/TanhGradAgradients/rnn/basic_lstm_cell/Add_31_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_31_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_31_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_31_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_31_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_31_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_45_grad/ShapeShapernn/basic_lstm_cell/Add_29*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_45_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_45*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_45_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_45_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_45_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_45_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_45*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_45_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_45_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_45_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_45_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_45_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_45_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_45_grad/Mul_1Mulrnn/basic_lstm_cell/Add_29Bgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_45_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_45_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_45_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_45_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_45_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_46_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_46*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_46_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_30*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_46_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_46_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_46_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_46_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_30*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_46_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_46_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_46_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_46_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_46_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_46_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_46_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_46Dgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_46_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_46_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_46_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_46_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_46_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_45_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_45Dgradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_46_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_46Bgradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_30_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_30Dgradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_30_grad/ShapeShapernn/basic_lstm_cell/split_15:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_30_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_30_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_30_grad/Shape1gradients/rnn/basic_lstm_cell/Add_30_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_30_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_45_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_30_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_30_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_30_grad/Sum/gradients/rnn/basic_lstm_cell/Add_30_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_30_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_45_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_30_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_30_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_30_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_30_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_30_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_30_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_30_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_30_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape_1*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_15_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_46_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_30_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_30_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_47_grad/SigmoidGradrnn/basic_lstm_cell/Const_45*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_15_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_15_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_15_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_15_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/BiasAddGrad*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_15Fgradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_15_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_15_grad/modFloorMod"rnn/basic_lstm_cell/concat_15/axis1gradients/rnn/basic_lstm_cell/concat_15_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeShapesplit:15*
T0*
out_type0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeNShapeNsplit:15rnn/basic_lstm_cell/Mul_44*
T0*
out_type0*
N* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_15_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_15_grad/mod3gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_15_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_15_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_15_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_15_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_15_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_15_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_15_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_15_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_15_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_15_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_15_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_15_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_15_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_15_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_15_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_44_grad/ShapeShapernn/basic_lstm_cell/Tanh_29*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_44_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_44*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_44_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_44_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_44_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_44_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_15_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_44*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_44_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_44_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_44_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_44_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_44_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_44_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_44_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_29Ggradients/rnn/basic_lstm_cell/concat_15_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_44_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_44_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_44_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_44_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_44_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_29_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_29Bgradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_44_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_44Dgradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddNAddNBgradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_29_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_29_grad/ShapeShapernn/basic_lstm_cell/Mul_42*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_29_grad/Shape_1Shapernn/basic_lstm_cell/Mul_43*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_29_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_29_grad/Shape1gradients/rnn/basic_lstm_cell/Add_29_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Х
-gradients/rnn/basic_lstm_cell/Add_29_grad/SumSumgradients/AddN?gradients/rnn/basic_lstm_cell/Add_29_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_29_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_29_grad/Sum/gradients/rnn/basic_lstm_cell/Add_29_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Щ
/gradients/rnn/basic_lstm_cell/Add_29_grad/Sum_1Sumgradients/AddNAgradients/rnn/basic_lstm_cell/Add_29_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_29_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_29_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_29_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_29_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_29_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_42_grad/ShapeShapernn/basic_lstm_cell/Add_27*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_42_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_42*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_42_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_42_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_42_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_42_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_42*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_42_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_42_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_42_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_42_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_42_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_42_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_42_grad/Mul_1Mulrnn/basic_lstm_cell/Add_27Bgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_42_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_42_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_42_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_42_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_42_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_43_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_43*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_43_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_28*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_43_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_43_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_43_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_43_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_28*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_43_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_43_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_43_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_43_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_43_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_43_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_43_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_43Dgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_43_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_43_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_43_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_43_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_43_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_42_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_42Dgradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_43_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_43Bgradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_28_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_28Dgradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_28_grad/ShapeShapernn/basic_lstm_cell/split_14:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_28_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_28_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_28_grad/Shape1gradients/rnn/basic_lstm_cell/Add_28_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_28_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_42_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_28_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_28_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_28_grad/Sum/gradients/rnn/basic_lstm_cell/Add_28_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_28_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_42_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_28_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_28_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_28_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_28_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_28_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_28_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_28_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_28_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape_1*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_14_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_43_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_28_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_28_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_44_grad/SigmoidGradrnn/basic_lstm_cell/Const_42*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_14_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_14_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_14_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_14_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/BiasAddGrad*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_14Fgradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_14_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_14_grad/modFloorMod"rnn/basic_lstm_cell/concat_14/axis1gradients/rnn/basic_lstm_cell/concat_14_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeShapesplit:14*
T0*
out_type0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeNShapeNsplit:14rnn/basic_lstm_cell/Mul_41*
out_type0*
N*
T0* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_14_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_14_grad/mod3gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_14_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_14_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_14_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_14_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_14_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_14_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_14_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_14_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_14_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_14_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_14_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_14_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_14_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_14_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_14_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_41_grad/ShapeShapernn/basic_lstm_cell/Tanh_27*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_41_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_41*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_41_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_41_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_41_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_41_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_14_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_41*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_41_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_41_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_41_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_41_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_41_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_41_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_41_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_27Ggradients/rnn/basic_lstm_cell/concat_14_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_41_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_41_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_41_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_41_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_41_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_27_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_27Bgradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_41_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_41Dgradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_1AddNBgradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_27_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_27_grad/ShapeShapernn/basic_lstm_cell/Mul_39*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_27_grad/Shape_1Shapernn/basic_lstm_cell/Mul_40*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_27_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_27_grad/Shape1gradients/rnn/basic_lstm_cell/Add_27_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_27_grad/SumSumgradients/AddN_1?gradients/rnn/basic_lstm_cell/Add_27_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_27_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_27_grad/Sum/gradients/rnn/basic_lstm_cell/Add_27_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_27_grad/Sum_1Sumgradients/AddN_1Agradients/rnn/basic_lstm_cell/Add_27_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_27_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_27_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_27_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_27_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_27_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_39_grad/ShapeShapernn/basic_lstm_cell/Add_25*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_39_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_39*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_39_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_39_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_39_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_39_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_39*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_39_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_39_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_39_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_39_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_39_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_39_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_39_grad/Mul_1Mulrnn/basic_lstm_cell/Add_25Bgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_39_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_39_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_39_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_39_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_39_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_40_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_40*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_40_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_26*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_40_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_40_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_40_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_40_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_26*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_40_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_40_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_40_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_40_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_40_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_40_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_40_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_40Dgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_40_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_40_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_40_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_40_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_40_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_39_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_39Dgradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_40_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_40Bgradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_26_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_26Dgradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_26_grad/ShapeShapernn/basic_lstm_cell/split_13:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_26_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_26_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_26_grad/Shape1gradients/rnn/basic_lstm_cell/Add_26_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_26_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_39_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_26_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_26_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_26_grad/Sum/gradients/rnn/basic_lstm_cell/Add_26_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_26_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_39_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_26_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_26_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_26_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_26_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_26_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_26_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_26_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_26_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape_1*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_13_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_40_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_26_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_26_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_41_grad/SigmoidGradrnn/basic_lstm_cell/Const_39*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_13_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_13_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_13_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_13_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/group_deps*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/BiasAddGrad*
T0*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_13Fgradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_13_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_13_grad/modFloorMod"rnn/basic_lstm_cell/concat_13/axis1gradients/rnn/basic_lstm_cell/concat_13_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeShapesplit:13*
out_type0*
T0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeNShapeNsplit:13rnn/basic_lstm_cell/Mul_38*
T0*
out_type0*
N* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_13_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_13_grad/mod3gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_13_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_13_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_13_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_13_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_13_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_13_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_13_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_13_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_13_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_13_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_13_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_13_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_13_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_13_grad/tuple/group_deps*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_13_grad/Slice_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_38_grad/ShapeShapernn/basic_lstm_cell/Tanh_25*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_38_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_38*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_38_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_38_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_38_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_38_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_13_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_38*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_38_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_38_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_38_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_38_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_38_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_38_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_38_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_25Ggradients/rnn/basic_lstm_cell/concat_13_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_38_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_38_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_38_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_38_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_38_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_25_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_25Bgradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_38_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_38Dgradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_2AddNBgradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_25_grad/TanhGrad*
N*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_25_grad/ShapeShapernn/basic_lstm_cell/Mul_36*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_25_grad/Shape_1Shapernn/basic_lstm_cell/Mul_37*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_25_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_25_grad/Shape1gradients/rnn/basic_lstm_cell/Add_25_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_25_grad/SumSumgradients/AddN_2?gradients/rnn/basic_lstm_cell/Add_25_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_25_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_25_grad/Sum/gradients/rnn/basic_lstm_cell/Add_25_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_25_grad/Sum_1Sumgradients/AddN_2Agradients/rnn/basic_lstm_cell/Add_25_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_25_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_25_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_25_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_25_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_25_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_36_grad/ShapeShapernn/basic_lstm_cell/Add_23*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_36_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_36*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_36_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_36_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_36_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_36_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_36*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_36_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_36_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_36_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_36_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_36_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_36_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_36_grad/Mul_1Mulrnn/basic_lstm_cell/Add_23Bgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_36_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_36_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_36_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_36_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_36_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_37_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_37*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_37_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_24*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_37_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_37_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_37_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_37_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_24*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_37_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_37_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_37_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_37_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_37_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_37_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_37_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_37Dgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_37_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_37_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_37_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_37_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_37_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_36_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_36Dgradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_37_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_37Bgradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_24_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_24Dgradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_24_grad/ShapeShapernn/basic_lstm_cell/split_12:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_24_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_24_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_24_grad/Shape1gradients/rnn/basic_lstm_cell/Add_24_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_24_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_36_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_24_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_24_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_24_grad/Sum/gradients/rnn/basic_lstm_cell/Add_24_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_24_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_36_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_24_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_24_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_24_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_24_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_24_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_24_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_24_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_24_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape_1*
T0*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_12_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_37_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_24_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_24_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_38_grad/SigmoidGradrnn/basic_lstm_cell/Const_36*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_12_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_12_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_12_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_12_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/BiasAddGrad*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_12Fgradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_12_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_12_grad/modFloorMod"rnn/basic_lstm_cell/concat_12/axis1gradients/rnn/basic_lstm_cell/concat_12_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeShapesplit:12*
T0*
out_type0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeNShapeNsplit:12rnn/basic_lstm_cell/Mul_35*
T0*
out_type0*
N* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_12_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_12_grad/mod3gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_12_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_12_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_12_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_12_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_12_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_12_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_12_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_12_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_12_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_12_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_12_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_12_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_12_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_12_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_12_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_35_grad/ShapeShapernn/basic_lstm_cell/Tanh_23*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_35_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_35*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_35_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_35_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_35_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_35_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_12_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_35*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_35_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_35_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_35_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_35_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_35_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_35_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_35_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_23Ggradients/rnn/basic_lstm_cell/concat_12_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_35_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_35_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_35_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_35_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_35_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_23_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_23Bgradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_35_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_35Dgradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_3AddNBgradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_23_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_23_grad/ShapeShapernn/basic_lstm_cell/Mul_33*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_23_grad/Shape_1Shapernn/basic_lstm_cell/Mul_34*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_23_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_23_grad/Shape1gradients/rnn/basic_lstm_cell/Add_23_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_23_grad/SumSumgradients/AddN_3?gradients/rnn/basic_lstm_cell/Add_23_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_23_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_23_grad/Sum/gradients/rnn/basic_lstm_cell/Add_23_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_23_grad/Sum_1Sumgradients/AddN_3Agradients/rnn/basic_lstm_cell/Add_23_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_23_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_23_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_23_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_23_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_23_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_33_grad/ShapeShapernn/basic_lstm_cell/Add_21*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_33_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_33*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_33_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_33_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_33_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_33_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_33*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_33_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_33_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_33_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_33_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_33_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_33_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_33_grad/Mul_1Mulrnn/basic_lstm_cell/Add_21Bgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_33_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_33_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_33_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_33_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_33_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_34_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_34*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_34_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_22*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_34_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_34_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_34_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_34_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_22*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_34_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_34_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_34_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_34_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_34_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_34_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_34_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_34Dgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_34_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_34_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_34_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_34_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_34_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_33_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_33Dgradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_34_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_34Bgradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_22_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_22Dgradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_22_grad/ShapeShapernn/basic_lstm_cell/split_11:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_22_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_22_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_22_grad/Shape1gradients/rnn/basic_lstm_cell/Add_22_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_22_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_33_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_22_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_22_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_22_grad/Sum/gradients/rnn/basic_lstm_cell/Add_22_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_22_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_33_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_22_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_22_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_22_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_22_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_22_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_22_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_22_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_22_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape_1*
T0*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_11_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_34_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_22_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_22_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_35_grad/SigmoidGradrnn/basic_lstm_cell/Const_33*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_11_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_11_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_11_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_11_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/BiasAddGrad*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_11Fgradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_11_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_11_grad/modFloorMod"rnn/basic_lstm_cell/concat_11/axis1gradients/rnn/basic_lstm_cell/concat_11_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeShapesplit:11*
T0*
out_type0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeNShapeNsplit:11rnn/basic_lstm_cell/Mul_32*
T0*
out_type0*
N* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_11_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_11_grad/mod3gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_11_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_11_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_11_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_11_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_11_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_11_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_11_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_11_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_11_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_11_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_11_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_11_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_11_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_11_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_11_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_32_grad/ShapeShapernn/basic_lstm_cell/Tanh_21*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_32_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_32*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_32_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_32_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_32_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_32_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_11_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_32*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_32_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_32_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_32_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_32_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_32_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_32_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_32_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_21Ggradients/rnn/basic_lstm_cell/concat_11_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_32_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_32_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_32_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_32_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_32_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_21_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_21Bgradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_32_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_32Dgradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_4AddNBgradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_21_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_21_grad/ShapeShapernn/basic_lstm_cell/Mul_30*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_21_grad/Shape_1Shapernn/basic_lstm_cell/Mul_31*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_21_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_21_grad/Shape1gradients/rnn/basic_lstm_cell/Add_21_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_21_grad/SumSumgradients/AddN_4?gradients/rnn/basic_lstm_cell/Add_21_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_21_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_21_grad/Sum/gradients/rnn/basic_lstm_cell/Add_21_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_21_grad/Sum_1Sumgradients/AddN_4Agradients/rnn/basic_lstm_cell/Add_21_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_21_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_21_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_21_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_21_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_21_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_30_grad/ShapeShapernn/basic_lstm_cell/Add_19*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_30_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_30*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_30_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_30_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_30_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_30_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_30*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_30_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_30_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_30_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_30_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_30_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_30_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_30_grad/Mul_1Mulrnn/basic_lstm_cell/Add_19Bgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_30_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_30_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_30_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_30_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_30_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_31_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_31*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_31_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_20*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_31_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_31_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_31_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_31_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_20*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_31_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_31_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_31_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_31_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_31_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_31_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_31_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_31Dgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_31_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_31_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_31_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_31_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_31_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_30_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_30Dgradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_31_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_31Bgradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_20_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_20Dgradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_20_grad/ShapeShapernn/basic_lstm_cell/split_10:2*
out_type0*
T0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_20_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_20_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_20_grad/Shape1gradients/rnn/basic_lstm_cell/Add_20_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_20_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_30_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_20_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_20_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_20_grad/Sum/gradients/rnn/basic_lstm_cell/Add_20_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_20_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_30_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_20_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_20_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_20_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_20_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_20_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_20_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_20_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_20_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape_1*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_10_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_31_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_20_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_20_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_32_grad/SigmoidGradrnn/basic_lstm_cell/Const_30*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_10_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_10_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_10_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_10_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/BiasAddGrad*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_10Fgradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_10_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_10_grad/modFloorMod"rnn/basic_lstm_cell/concat_10/axis1gradients/rnn/basic_lstm_cell/concat_10_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeShapesplit:10*
out_type0*
T0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeNShapeNsplit:10rnn/basic_lstm_cell/Mul_29*
T0*
out_type0*
N* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_10_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_10_grad/mod3gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_10_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_10_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_10_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_10_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_10_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_10_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_10_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_10_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_10_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_10_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_10_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_10_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_10_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_10_grad/tuple/group_deps*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_10_grad/Slice_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_29_grad/ShapeShapernn/basic_lstm_cell/Tanh_19*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_29_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_29*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_29_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_29_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_29_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_29_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_10_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_29*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_29_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_29_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_29_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_29_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_29_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_29_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_29_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_19Ggradients/rnn/basic_lstm_cell/concat_10_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_29_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_29_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_29_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_29_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_29_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_19_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_19Bgradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_29_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_29Dgradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_5AddNBgradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_19_grad/TanhGrad*
N*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_19_grad/ShapeShapernn/basic_lstm_cell/Mul_27*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_19_grad/Shape_1Shapernn/basic_lstm_cell/Mul_28*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_19_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_19_grad/Shape1gradients/rnn/basic_lstm_cell/Add_19_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_19_grad/SumSumgradients/AddN_5?gradients/rnn/basic_lstm_cell/Add_19_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_19_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_19_grad/Sum/gradients/rnn/basic_lstm_cell/Add_19_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_19_grad/Sum_1Sumgradients/AddN_5Agradients/rnn/basic_lstm_cell/Add_19_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_19_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_19_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_19_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_19_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_19_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_27_grad/ShapeShapernn/basic_lstm_cell/Add_17*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_27_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_27*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_27_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_27_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_27_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_27_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_27*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_27_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_27_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_27_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_27_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_27_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_27_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_27_grad/Mul_1Mulrnn/basic_lstm_cell/Add_17Bgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_27_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_27_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_27_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_27_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_27_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_28_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_28*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_28_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_18*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_28_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_28_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_28_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_28_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_18*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_28_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_28_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_28_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_28_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_28_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_28_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_28_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_28Dgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_28_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_28_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_28_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_28_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_28_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_27_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_27Dgradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_28_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_28Bgradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_18_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_18Dgradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_18_grad/ShapeShapernn/basic_lstm_cell/split_9:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_18_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_18_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_18_grad/Shape1gradients/rnn/basic_lstm_cell/Add_18_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_18_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_27_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_18_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_18_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_18_grad/Sum/gradients/rnn/basic_lstm_cell/Add_18_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_18_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_27_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_18_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_18_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_18_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_18_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_18_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_18_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_18_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_18_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_9_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_28_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_18_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_18_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_29_grad/SigmoidGradrnn/basic_lstm_cell/Const_27*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_9_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_9_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_9_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_9_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_9Egradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_9_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_9_grad/modFloorMod!rnn/basic_lstm_cell/concat_9/axis0gradients/rnn/basic_lstm_cell/concat_9_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeShapesplit:9*
out_type0*
T0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeNShapeNsplit:9rnn/basic_lstm_cell/Mul_26*
N*
T0*
out_type0* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_9_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_9_grad/mod2gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_9_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_9_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_9_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_9_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_9_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_9_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_9_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_9_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_9_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_9_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_9_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_9_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_9_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_26_grad/ShapeShapernn/basic_lstm_cell/Tanh_17*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_26_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_26*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_26_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_26_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_26_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_26_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_26*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_26_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_26_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_26_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_26_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_26_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_26_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_26_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_17Fgradients/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_26_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_26_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_26_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_26_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_26_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_17_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_17Bgradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_26_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_26Dgradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_6AddNBgradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_17_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_17_grad/ShapeShapernn/basic_lstm_cell/Mul_24*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_17_grad/Shape_1Shapernn/basic_lstm_cell/Mul_25*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_17_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_17_grad/Shape1gradients/rnn/basic_lstm_cell/Add_17_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_17_grad/SumSumgradients/AddN_6?gradients/rnn/basic_lstm_cell/Add_17_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_17_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_17_grad/Sum/gradients/rnn/basic_lstm_cell/Add_17_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_17_grad/Sum_1Sumgradients/AddN_6Agradients/rnn/basic_lstm_cell/Add_17_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_17_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_17_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_17_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_17_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_17_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_24_grad/ShapeShapernn/basic_lstm_cell/Add_15*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_24_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_24*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_24_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_24_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_24_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_24_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_24*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_24_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_24_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_24_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_24_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_24_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_24_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_24_grad/Mul_1Mulrnn/basic_lstm_cell/Add_15Bgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_24_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_24_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_24_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_24_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_24_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_25_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_25*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_25_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_16*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_25_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_25_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_25_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_25_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_16*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_25_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_25_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_25_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_25_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_25_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_25_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_25_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_25Dgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_25_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_25_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_25_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_25_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_25_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_24_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_24Dgradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_25_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_25Bgradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_16_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_16Dgradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_16_grad/ShapeShapernn/basic_lstm_cell/split_8:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_16_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_16_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_16_grad/Shape1gradients/rnn/basic_lstm_cell/Add_16_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_16_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_24_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_16_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_16_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_16_grad/Sum/gradients/rnn/basic_lstm_cell/Add_16_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_16_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_24_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_16_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_16_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_16_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_16_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_16_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_16_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_16_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_16_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_8_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_25_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_16_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_16_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_26_grad/SigmoidGradrnn/basic_lstm_cell/Const_24*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_8_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_8_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_8_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_8_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_8Egradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_8_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_8_grad/modFloorMod!rnn/basic_lstm_cell/concat_8/axis0gradients/rnn/basic_lstm_cell/concat_8_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeShapesplit:8*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeNShapeNsplit:8rnn/basic_lstm_cell/Mul_23*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_8_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_8_grad/mod2gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_8_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_8_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_8_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_8_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_8_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_8_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_8_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_8_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_8_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_8_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_8_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_8_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_8_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_23_grad/ShapeShapernn/basic_lstm_cell/Tanh_15*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_23_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_23*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_23_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_23_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_23_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_23_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_23*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_23_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_23_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_23_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_23_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_23_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_23_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_23_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_15Fgradients/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_23_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_23_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_23_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_23_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_23_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_15_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_15Bgradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_23_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_23Dgradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_7AddNBgradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_15_grad/TanhGrad*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape*
N*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_15_grad/ShapeShapernn/basic_lstm_cell/Mul_21*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_15_grad/Shape_1Shapernn/basic_lstm_cell/Mul_22*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_15_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_15_grad/Shape1gradients/rnn/basic_lstm_cell/Add_15_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_15_grad/SumSumgradients/AddN_7?gradients/rnn/basic_lstm_cell/Add_15_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_15_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_15_grad/Sum/gradients/rnn/basic_lstm_cell/Add_15_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_15_grad/Sum_1Sumgradients/AddN_7Agradients/rnn/basic_lstm_cell/Add_15_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_15_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_15_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_15_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_15_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_15_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_21_grad/ShapeShapernn/basic_lstm_cell/Add_13*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_21_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_21*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_21_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_21_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_21_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_21_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_21*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_21_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_21_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_21_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_21_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_21_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_21_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_21_grad/Mul_1Mulrnn/basic_lstm_cell/Add_13Bgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_21_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_21_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_21_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_21_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_21_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_22_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_22*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_22_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_14*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_22_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_22_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_22_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_22_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_14*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_22_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_22_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_22_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_22_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_22_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_22_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_22_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_22Dgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_22_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_22_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_22_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_22_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_22_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_21_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_21Dgradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_22_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_22Bgradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_14_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_14Dgradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_14_grad/ShapeShapernn/basic_lstm_cell/split_7:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_14_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_14_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_14_grad/Shape1gradients/rnn/basic_lstm_cell/Add_14_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_14_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_21_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_14_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_14_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_14_grad/Sum/gradients/rnn/basic_lstm_cell/Add_14_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_14_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_21_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_14_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_14_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_14_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_14_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_14_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_14_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_14_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_14_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_7_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_22_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_14_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_14_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_23_grad/SigmoidGradrnn/basic_lstm_cell/Const_21*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_7_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_7_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_7_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_7_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_7Egradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_7_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_7_grad/modFloorMod!rnn/basic_lstm_cell/concat_7/axis0gradients/rnn/basic_lstm_cell/concat_7_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeShapesplit:7*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeNShapeNsplit:7rnn/basic_lstm_cell/Mul_20*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_7_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_7_grad/mod2gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_7_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_7_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_7_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_7_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_7_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_7_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_7_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_7_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_7_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_7_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_7_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_7_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_7_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_20_grad/ShapeShapernn/basic_lstm_cell/Tanh_13*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_20_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_20*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_20_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_20_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_20_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_20_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_20*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_20_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_20_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_20_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_20_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_20_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_20_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_20_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_13Fgradients/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_20_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_20_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_20_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_20_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_20_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_13_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_13Bgradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_20_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_20Dgradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_8AddNBgradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_13_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_13_grad/ShapeShapernn/basic_lstm_cell/Mul_18*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_13_grad/Shape_1Shapernn/basic_lstm_cell/Mul_19*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_13_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_13_grad/Shape1gradients/rnn/basic_lstm_cell/Add_13_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_13_grad/SumSumgradients/AddN_8?gradients/rnn/basic_lstm_cell/Add_13_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_13_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_13_grad/Sum/gradients/rnn/basic_lstm_cell/Add_13_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_13_grad/Sum_1Sumgradients/AddN_8Agradients/rnn/basic_lstm_cell/Add_13_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_13_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_13_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_13_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_13_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_13_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_18_grad/ShapeShapernn/basic_lstm_cell/Add_11*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_18_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_18*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_18_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_18_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_18_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_18_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_18*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_18_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_18_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_18_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_18_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_18_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_18_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_18_grad/Mul_1Mulrnn/basic_lstm_cell/Add_11Bgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_18_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_18_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_18_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_18_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_18_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_19_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_19*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_19_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_12*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_19_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_19_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_19_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_19_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_12*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_19_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_19_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_19_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_19_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_19_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_19_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_19_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_19Dgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_19_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_19_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_19_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_19_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_19_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_18_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_18Dgradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_19_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_19Bgradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_12_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_12Dgradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_12_grad/ShapeShapernn/basic_lstm_cell/split_6:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_12_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_12_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_12_grad/Shape1gradients/rnn/basic_lstm_cell/Add_12_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_12_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_18_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_12_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_12_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_12_grad/Sum/gradients/rnn/basic_lstm_cell/Add_12_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_12_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_18_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_12_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_12_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_12_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_12_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_12_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_12_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_12_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_12_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_6_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_19_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_12_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_12_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_20_grad/SigmoidGradrnn/basic_lstm_cell/Const_18*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_6_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_6_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_6_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_6_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_6Egradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_6_grad/RankConst*
dtype0*
value	B :*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_6_grad/modFloorMod!rnn/basic_lstm_cell/concat_6/axis0gradients/rnn/basic_lstm_cell/concat_6_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeShapesplit:6*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeNShapeNsplit:6rnn/basic_lstm_cell/Mul_17*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_6_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_6_grad/mod2gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_6_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_6_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_6_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_6_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_6_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_6_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_6_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_6_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_6_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_6_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_6_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_6_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_6_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_17_grad/ShapeShapernn/basic_lstm_cell/Tanh_11*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_17_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_17*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_17_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_17_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_17_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_17_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_17*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_17_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_17_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_17_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_17_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_17_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_17_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_17_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_11Fgradients/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_17_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_17_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_17_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_17_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_17_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_11_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_11Bgradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_17_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_17Dgradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_9AddNBgradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_11_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_11_grad/ShapeShapernn/basic_lstm_cell/Mul_15*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_11_grad/Shape_1Shapernn/basic_lstm_cell/Mul_16*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_11_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_11_grad/Shape1gradients/rnn/basic_lstm_cell/Add_11_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_11_grad/SumSumgradients/AddN_9?gradients/rnn/basic_lstm_cell/Add_11_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_11_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_11_grad/Sum/gradients/rnn/basic_lstm_cell/Add_11_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_11_grad/Sum_1Sumgradients/AddN_9Agradients/rnn/basic_lstm_cell/Add_11_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_11_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_11_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_11_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_11_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_11_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_15_grad/ShapeShapernn/basic_lstm_cell/Add_9*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_15_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_15*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_15_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_15_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_15_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_15_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_15*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_15_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_15_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_15_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_15_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_15_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_15_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ч
/gradients/rnn/basic_lstm_cell/Mul_15_grad/Mul_1Mulrnn/basic_lstm_cell/Add_9Bgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_15_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_15_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_15_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_15_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_15_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_16_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_16*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_16_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_10*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_16_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_16_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_16_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_16_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_10*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_16_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_16_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_16_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_16_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_16_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_16_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_16_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_16Dgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_16_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_16_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_16_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_16_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_16_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_15_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_15Dgradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_16_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_16Bgradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_10_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_10Dgradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_10_grad/ShapeShapernn/basic_lstm_cell/split_5:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_10_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_10_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_10_grad/Shape1gradients/rnn/basic_lstm_cell/Add_10_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_10_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_15_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_10_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_10_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_10_grad/Sum/gradients/rnn/basic_lstm_cell/Add_10_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_10_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_15_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_10_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_10_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_10_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_10_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_10_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_10_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_10_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_10_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_5_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_16_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_10_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_10_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_17_grad/SigmoidGradrnn/basic_lstm_cell/Const_15*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_5_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_5_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_5_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_5_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/group_deps*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGrad*
T0*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_5Egradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_5_grad/RankConst*
dtype0*
value	B :*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_5_grad/modFloorMod!rnn/basic_lstm_cell/concat_5/axis0gradients/rnn/basic_lstm_cell/concat_5_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeShapesplit:5*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeNShapeNsplit:5rnn/basic_lstm_cell/Mul_14*
N*
T0*
out_type0* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_5_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_5_grad/mod2gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_5_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_5_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_5_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_5_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_5_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_5_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_5_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_5_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_5_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_5_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_5_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_5_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_5_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_14_grad/ShapeShapernn/basic_lstm_cell/Tanh_9*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_14_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_14*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_14_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_14_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_14_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_14_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_14*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_14_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_14_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_14_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_14_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_14_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_14_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ь
/gradients/rnn/basic_lstm_cell/Mul_14_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_9Fgradients/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_14_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_14_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_14_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_14_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_14_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
а
2gradients/rnn/basic_lstm_cell/Tanh_9_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_9Bgradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_14_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_14Dgradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_10AddNBgradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependency2gradients/rnn/basic_lstm_cell/Tanh_9_grad/TanhGrad*
N*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_9_grad/ShapeShapernn/basic_lstm_cell/Mul_12*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Add_9_grad/Shape_1Shapernn/basic_lstm_cell/Mul_13*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Add_9_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_9_grad/Shape0gradients/rnn/basic_lstm_cell/Add_9_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Add_9_grad/SumSumgradients/AddN_10>gradients/rnn/basic_lstm_cell/Add_9_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_9_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_9_grad/Sum.gradients/rnn/basic_lstm_cell/Add_9_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
.gradients/rnn/basic_lstm_cell/Add_9_grad/Sum_1Sumgradients/AddN_10@gradients/rnn/basic_lstm_cell/Add_9_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_9_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_9_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Add_9_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_9_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_9_grad/tuple/group_deps*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_12_grad/ShapeShapernn/basic_lstm_cell/Add_7*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_12_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_12*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_12_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_12_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_12_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_12_grad/MulMulAgradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_12*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_12_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_12_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_12_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_12_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_12_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_12_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
/gradients/rnn/basic_lstm_cell/Mul_12_grad/Mul_1Mulrnn/basic_lstm_cell/Add_7Agradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_12_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_12_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_12_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_12_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_12_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_13_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_13*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_13_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_8*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_13_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_13_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_13_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Mul_13_grad/MulMulCgradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_8*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_13_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_13_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_13_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_13_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_13_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_13_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_13_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_13Cgradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_13_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_13_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_13_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_13_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_13_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_12_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_12Dgradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_13_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_13Bgradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
в
2gradients/rnn/basic_lstm_cell/Tanh_8_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_8Dgradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_8_grad/ShapeShapernn/basic_lstm_cell/split_4:2*
T0*
out_type0*
_output_shapes
:
s
0gradients/rnn/basic_lstm_cell/Add_8_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
і
>gradients/rnn/basic_lstm_cell/Add_8_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_8_grad/Shape0gradients/rnn/basic_lstm_cell/Add_8_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ю
,gradients/rnn/basic_lstm_cell/Add_8_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_12_grad/SigmoidGrad>gradients/rnn/basic_lstm_cell/Add_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_8_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_8_grad/Sum.gradients/rnn/basic_lstm_cell/Add_8_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ђ
.gradients/rnn/basic_lstm_cell/Add_8_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_12_grad/SigmoidGrad@gradients/rnn/basic_lstm_cell/Add_8_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ю
2gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_8_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Љ
9gradients/rnn/basic_lstm_cell/Add_8_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_8_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_8_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ї
Cgradients/rnn/basic_lstm_cell/Add_8_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_8_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_4_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_13_grad/SigmoidGrad2gradients/rnn/basic_lstm_cell/Tanh_8_grad/TanhGradAgradients/rnn/basic_lstm_cell/Add_8_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_14_grad/SigmoidGradrnn/basic_lstm_cell/Const_12*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_4_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_4_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_4_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_4_grad/concat*
T0*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_4Egradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_4_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_4_grad/modFloorMod!rnn/basic_lstm_cell/concat_4/axis0gradients/rnn/basic_lstm_cell/concat_4_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeShapesplit:4*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeNShapeNsplit:4rnn/basic_lstm_cell/Mul_11*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_4_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_4_grad/mod2gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_4_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_4_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_4_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_4_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_4_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_4_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_4_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_4_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_4_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_4_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_4_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_4_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_4_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_11_grad/ShapeShapernn/basic_lstm_cell/Tanh_7*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_11_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_11*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_11_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_11_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_11_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_11_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_11*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_11_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_11_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_11_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_11_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_11_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_11_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ь
/gradients/rnn/basic_lstm_cell/Mul_11_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_7Fgradients/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_11_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_11_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_11_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_11_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_11_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
а
2gradients/rnn/basic_lstm_cell/Tanh_7_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_7Bgradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_11_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_11Dgradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_11AddNBgradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependency2gradients/rnn/basic_lstm_cell/Tanh_7_grad/TanhGrad*
N*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_7_grad/ShapeShapernn/basic_lstm_cell/Mul_9*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Add_7_grad/Shape_1Shapernn/basic_lstm_cell/Mul_10*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Add_7_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_7_grad/Shape0gradients/rnn/basic_lstm_cell/Add_7_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Add_7_grad/SumSumgradients/AddN_11>gradients/rnn/basic_lstm_cell/Add_7_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_7_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_7_grad/Sum.gradients/rnn/basic_lstm_cell/Add_7_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
.gradients/rnn/basic_lstm_cell/Add_7_grad/Sum_1Sumgradients/AddN_11@gradients/rnn/basic_lstm_cell/Add_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_7_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_7_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Add_7_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_7_grad/tuple/group_deps*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_7_grad/tuple/group_deps*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_9_grad/ShapeShapernn/basic_lstm_cell/Add_5*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_9_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_9*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_9_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_9_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_9_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
,gradients/rnn/basic_lstm_cell/Mul_9_grad/MulMulAgradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_9*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_9_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_9_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_9_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_9_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_9_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_9_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Х
.gradients/rnn/basic_lstm_cell/Mul_9_grad/Mul_1Mulrnn/basic_lstm_cell/Add_5Agradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_9_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_9_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_9_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_9_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_9_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_10_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_10*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_10_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_6*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_10_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_10_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_10_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Mul_10_grad/MulMulCgradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_6*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_10_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_10_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_10_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_10_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_10_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_10_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_10_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_10Cgradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_10_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_10_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_10_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_10_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_10_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_9_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_9Cgradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_10_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_10Bgradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
в
2gradients/rnn/basic_lstm_cell/Tanh_6_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_6Dgradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_6_grad/ShapeShapernn/basic_lstm_cell/split_3:2*
T0*
out_type0*
_output_shapes
:
s
0gradients/rnn/basic_lstm_cell/Add_6_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
і
>gradients/rnn/basic_lstm_cell/Add_6_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_6_grad/Shape0gradients/rnn/basic_lstm_cell/Add_6_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
э
,gradients/rnn/basic_lstm_cell/Add_6_grad/SumSum8gradients/rnn/basic_lstm_cell/Sigmoid_9_grad/SigmoidGrad>gradients/rnn/basic_lstm_cell/Add_6_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_6_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_6_grad/Sum.gradients/rnn/basic_lstm_cell/Add_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ё
.gradients/rnn/basic_lstm_cell/Add_6_grad/Sum_1Sum8gradients/rnn/basic_lstm_cell/Sigmoid_9_grad/SigmoidGrad@gradients/rnn/basic_lstm_cell/Add_6_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ю
2gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_6_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Љ
9gradients/rnn/basic_lstm_cell/Add_6_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_6_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_6_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ї
Cgradients/rnn/basic_lstm_cell/Add_6_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_6_grad/tuple/group_deps*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape_1*
T0*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_3_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_10_grad/SigmoidGrad2gradients/rnn/basic_lstm_cell/Tanh_6_grad/TanhGradAgradients/rnn/basic_lstm_cell/Add_6_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_11_grad/SigmoidGradrnn/basic_lstm_cell/Const_9*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_3_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_3_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_3_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_3_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_3Egradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/group_deps*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_3_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_3_grad/modFloorMod!rnn/basic_lstm_cell/concat_3/axis0gradients/rnn/basic_lstm_cell/concat_3_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeShapesplit:3*
T0*
out_type0*
_output_shapes
:
Є
2gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeNShapeNsplit:3rnn/basic_lstm_cell/Mul_8*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_3_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_3_grad/mod2gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_3_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_3_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_3_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_3_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_3_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_3_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_3_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_3_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_3_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_3_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_3_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_3_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_8_grad/ShapeShapernn/basic_lstm_cell/Tanh_5*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_8_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_8*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_8_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_8_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_8_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ь
,gradients/rnn/basic_lstm_cell/Mul_8_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_8*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_8_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_8_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_8_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_8_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_8_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_8_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_5Fgradients/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_8_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_8_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_8_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_8_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_8_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Я
2gradients/rnn/basic_lstm_cell/Tanh_5_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_5Agradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_8_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_8Cgradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_12AddNAgradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependency2gradients/rnn/basic_lstm_cell/Tanh_5_grad/TanhGrad*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_5_grad/ShapeShapernn/basic_lstm_cell/Mul_6*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Add_5_grad/Shape_1Shapernn/basic_lstm_cell/Mul_7*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Add_5_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_5_grad/Shape0gradients/rnn/basic_lstm_cell/Add_5_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Add_5_grad/SumSumgradients/AddN_12>gradients/rnn/basic_lstm_cell/Add_5_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_5_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_5_grad/Sum.gradients/rnn/basic_lstm_cell/Add_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
.gradients/rnn/basic_lstm_cell/Add_5_grad/Sum_1Sumgradients/AddN_12@gradients/rnn/basic_lstm_cell/Add_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_5_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_5_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Add_5_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_5_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_5_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_6_grad/ShapeShapernn/basic_lstm_cell/Add_3*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_6_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_6*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_6_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_6_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_6_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
,gradients/rnn/basic_lstm_cell/Mul_6_grad/MulMulAgradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_6*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_6_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_6_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_6_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_6_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_6_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Х
.gradients/rnn/basic_lstm_cell/Mul_6_grad/Mul_1Mulrnn/basic_lstm_cell/Add_3Agradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_6_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_6_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_6_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_6_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_6_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_7_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_7*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_7_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_4*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_7_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_7_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_7_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Mul_7_grad/MulMulCgradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_4*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_7_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_7_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_7_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_7_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_7_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_7_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_7_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_7Cgradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_7_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_7_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_7_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_7_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_6_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_6Cgradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
л
8gradients/rnn/basic_lstm_cell/Sigmoid_7_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_7Agradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
б
2gradients/rnn/basic_lstm_cell/Tanh_4_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_4Cgradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_4_grad/ShapeShapernn/basic_lstm_cell/split_2:2*
T0*
out_type0*
_output_shapes
:
s
0gradients/rnn/basic_lstm_cell/Add_4_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
і
>gradients/rnn/basic_lstm_cell/Add_4_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_4_grad/Shape0gradients/rnn/basic_lstm_cell/Add_4_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
э
,gradients/rnn/basic_lstm_cell/Add_4_grad/SumSum8gradients/rnn/basic_lstm_cell/Sigmoid_6_grad/SigmoidGrad>gradients/rnn/basic_lstm_cell/Add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_4_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_4_grad/Sum.gradients/rnn/basic_lstm_cell/Add_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ё
.gradients/rnn/basic_lstm_cell/Add_4_grad/Sum_1Sum8gradients/rnn/basic_lstm_cell/Sigmoid_6_grad/SigmoidGrad@gradients/rnn/basic_lstm_cell/Add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Ю
2gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_4_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Љ
9gradients/rnn/basic_lstm_cell/Add_4_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_4_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_4_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ї
Cgradients/rnn/basic_lstm_cell/Add_4_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_2_grad/concatConcatV28gradients/rnn/basic_lstm_cell/Sigmoid_7_grad/SigmoidGrad2gradients/rnn/basic_lstm_cell/Tanh_4_grad/TanhGradAgradients/rnn/basic_lstm_cell/Add_4_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/Sigmoid_8_grad/SigmoidGradrnn/basic_lstm_cell/Const_6*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_2_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_2_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_2_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_2_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_2Egradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_2_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_2_grad/modFloorMod!rnn/basic_lstm_cell/concat_2/axis0gradients/rnn/basic_lstm_cell/concat_2_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeShapesplit:2*
T0*
out_type0*
_output_shapes
:
Є
2gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeNShapeNsplit:2rnn/basic_lstm_cell/Mul_5*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_2_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_2_grad/mod2gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_2_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_2_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_2_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_2_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_2_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_2_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_2_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_2_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_2_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_2_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_2_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_2_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_5_grad/ShapeShapernn/basic_lstm_cell/Tanh_3*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_5_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_5*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_5_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_5_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_5_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ь
,gradients/rnn/basic_lstm_cell/Mul_5_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_5*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_5_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_5_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_5_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_5_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_5_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_5_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_3Fgradients/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_5_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_5_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_5_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_5_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_5_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Я
2gradients/rnn/basic_lstm_cell/Tanh_3_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_3Agradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_5_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_5Cgradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_13AddNAgradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependency2gradients/rnn/basic_lstm_cell/Tanh_3_grad/TanhGrad*
N*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_3_grad/ShapeShapernn/basic_lstm_cell/Mul_3*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Add_3_grad/Shape_1Shapernn/basic_lstm_cell/Mul_4*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_3_grad/Shape0gradients/rnn/basic_lstm_cell/Add_3_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Add_3_grad/SumSumgradients/AddN_13>gradients/rnn/basic_lstm_cell/Add_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_3_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_3_grad/Sum.gradients/rnn/basic_lstm_cell/Add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
.gradients/rnn/basic_lstm_cell/Add_3_grad/Sum_1Sumgradients/AddN_13@gradients/rnn/basic_lstm_cell/Add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_3_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_3_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Add_3_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_3_grad/ShapeShapernn/basic_lstm_cell/Add_1*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_3_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_3*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_3_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_3_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
,gradients/rnn/basic_lstm_cell/Mul_3_grad/MulMulAgradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_3*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_3_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_3_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_3_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_3_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Х
.gradients/rnn/basic_lstm_cell/Mul_3_grad/Mul_1Mulrnn/basic_lstm_cell/Add_1Agradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_3_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_3_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_3_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_3_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_4_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_4*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_4_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_2*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_4_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_4_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Mul_4_grad/MulMulCgradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_2*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_4_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_4_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_4_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_4_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_4_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_4_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_4Cgradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_4_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_4_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_4_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_4_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/group_deps*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_3Cgradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
л
8gradients/rnn/basic_lstm_cell/Sigmoid_4_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_4Agradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
б
2gradients/rnn/basic_lstm_cell/Tanh_2_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_2Cgradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_2_grad/ShapeShapernn/basic_lstm_cell/split_1:2*
out_type0*
T0*
_output_shapes
:
s
0gradients/rnn/basic_lstm_cell/Add_2_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
і
>gradients/rnn/basic_lstm_cell/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_2_grad/Shape0gradients/rnn/basic_lstm_cell/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
э
,gradients/rnn/basic_lstm_cell/Add_2_grad/SumSum8gradients/rnn/basic_lstm_cell/Sigmoid_3_grad/SigmoidGrad>gradients/rnn/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_2_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_2_grad/Sum.gradients/rnn/basic_lstm_cell/Add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ё
.gradients/rnn/basic_lstm_cell/Add_2_grad/Sum_1Sum8gradients/rnn/basic_lstm_cell/Sigmoid_3_grad/SigmoidGrad@gradients/rnn/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ю
2gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_2_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Љ
9gradients/rnn/basic_lstm_cell/Add_2_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_2_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_2_grad/tuple/group_deps*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
Ї
Cgradients/rnn/basic_lstm_cell/Add_2_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_1_grad/concatConcatV28gradients/rnn/basic_lstm_cell/Sigmoid_4_grad/SigmoidGrad2gradients/rnn/basic_lstm_cell/Tanh_2_grad/TanhGradAgradients/rnn/basic_lstm_cell/Add_2_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/Sigmoid_5_grad/SigmoidGradrnn/basic_lstm_cell/Const_3*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_1_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_1_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_1_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_1_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_1Egradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_1_grad/modFloorMod!rnn/basic_lstm_cell/concat_1/axis0gradients/rnn/basic_lstm_cell/concat_1_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeShapesplit:1*
T0*
out_type0*
_output_shapes
:
Є
2gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeNShapeNsplit:1rnn/basic_lstm_cell/Mul_2*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_1_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_1_grad/mod2gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_1_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_1_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_1_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_1_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_1_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_1_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_1_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_1_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_1_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_1_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_1_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_1_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_2_grad/ShapeShapernn/basic_lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_2_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_2_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ь
,gradients/rnn/basic_lstm_cell/Mul_2_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_2*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_2_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_2_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_2_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_2_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_2_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_1Fgradients/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_2_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_2_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_2_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Я
2gradients/rnn/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_1Agradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_2Cgradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_14AddNAgradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependency2gradients/rnn/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_1_grad/ShapeShapernn/basic_lstm_cell/Mul*
out_type0*
T0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Add_1_grad/Shape_1Shapernn/basic_lstm_cell/Mul_1*
out_type0*
T0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_1_grad/Shape0gradients/rnn/basic_lstm_cell/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Add_1_grad/SumSumgradients/AddN_14>gradients/rnn/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_1_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_1_grad/Sum.gradients/rnn/basic_lstm_cell/Add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
.gradients/rnn/basic_lstm_cell/Add_1_grad/Sum_1Sumgradients/AddN_14@gradients/rnn/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_1_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

,gradients/rnn/basic_lstm_cell/Mul_grad/ShapeShape rnn/BasicLSTMCellZeroState/zeros*
T0*
out_type0*
_output_shapes
:

.gradients/rnn/basic_lstm_cell/Mul_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
№
<gradients/rnn/basic_lstm_cell/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/rnn/basic_lstm_cell/Mul_grad/Shape.gradients/rnn/basic_lstm_cell/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
У
*gradients/rnn/basic_lstm_cell/Mul_grad/MulMulAgradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid*
T0*'
_output_shapes
:џџџџџџџџџ
л
*gradients/rnn/basic_lstm_cell/Mul_grad/SumSum*gradients/rnn/basic_lstm_cell/Mul_grad/Mul<gradients/rnn/basic_lstm_cell/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
г
.gradients/rnn/basic_lstm_cell/Mul_grad/ReshapeReshape*gradients/rnn/basic_lstm_cell/Mul_grad/Sum,gradients/rnn/basic_lstm_cell/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
,gradients/rnn/basic_lstm_cell/Mul_grad/Mul_1Mul rnn/BasicLSTMCellZeroState/zerosAgradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_grad/Sum_1Sum,gradients/rnn/basic_lstm_cell/Mul_grad/Mul_1>gradients/rnn/basic_lstm_cell/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_grad/Reshape_1Reshape,gradients/rnn/basic_lstm_cell/Mul_grad/Sum_1.gradients/rnn/basic_lstm_cell/Mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ѓ
7gradients/rnn/basic_lstm_cell/Mul_grad/tuple/group_depsNoOp/^gradients/rnn/basic_lstm_cell/Mul_grad/Reshape1^gradients/rnn/basic_lstm_cell/Mul_grad/Reshape_1
Њ
?gradients/rnn/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity.gradients/rnn/basic_lstm_cell/Mul_grad/Reshape8^gradients/rnn/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn/basic_lstm_cell/Mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
А
Agradients/rnn/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity0gradients/rnn/basic_lstm_cell/Mul_grad/Reshape_18^gradients/rnn/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_1_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_1_grad/Shape_1Shapernn/basic_lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_1_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ф
,gradients/rnn/basic_lstm_cell/Mul_1_grad/MulMulCgradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_1_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_1_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_1_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_1_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_1_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_1Cgradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_1_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_1_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_1_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
з
6gradients/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/SigmoidAgradients/rnn/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
л
8gradients/rnn/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_1Agradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
Э
0gradients/rnn/basic_lstm_cell/Tanh_grad/TanhGradTanhGradrnn/basic_lstm_cell/TanhCgradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

,gradients/rnn/basic_lstm_cell/Add_grad/ShapeShapernn/basic_lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:
q
.gradients/rnn/basic_lstm_cell/Add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
№
<gradients/rnn/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/rnn/basic_lstm_cell/Add_grad/Shape.gradients/rnn/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ч
*gradients/rnn/basic_lstm_cell/Add_grad/SumSum6gradients/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGrad<gradients/rnn/basic_lstm_cell/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
г
.gradients/rnn/basic_lstm_cell/Add_grad/ReshapeReshape*gradients/rnn/basic_lstm_cell/Add_grad/Sum,gradients/rnn/basic_lstm_cell/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ы
,gradients/rnn/basic_lstm_cell/Add_grad/Sum_1Sum6gradients/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGrad>gradients/rnn/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ш
0gradients/rnn/basic_lstm_cell/Add_grad/Reshape_1Reshape,gradients/rnn/basic_lstm_cell/Add_grad/Sum_1.gradients/rnn/basic_lstm_cell/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ѓ
7gradients/rnn/basic_lstm_cell/Add_grad/tuple/group_depsNoOp/^gradients/rnn/basic_lstm_cell/Add_grad/Reshape1^gradients/rnn/basic_lstm_cell/Add_grad/Reshape_1
Њ
?gradients/rnn/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity.gradients/rnn/basic_lstm_cell/Add_grad/Reshape8^gradients/rnn/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn/basic_lstm_cell/Add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

Agradients/rnn/basic_lstm_cell/Add_grad/tuple/control_dependency_1Identity0gradients/rnn/basic_lstm_cell/Add_grad/Reshape_18^gradients/rnn/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: 

/gradients/rnn/basic_lstm_cell/split_grad/concatConcatV28gradients/rnn/basic_lstm_cell/Sigmoid_1_grad/SigmoidGrad0gradients/rnn/basic_lstm_cell/Tanh_grad/TanhGrad?gradients/rnn/basic_lstm_cell/Add_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradrnn/basic_lstm_cell/Const*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџP
В
6gradients/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients/rnn/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Ў
;gradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOp7^gradients/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad0^gradients/rnn/basic_lstm_cell/split_grad/concat
Д
Cgradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity/gradients/rnn/basic_lstm_cell/split_grad/concat<^gradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn/basic_lstm_cell/split_grad/concat*'
_output_shapes
:џџџџџџџџџP
З
Egradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad<^gradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:P
ј
0gradients/rnn/basic_lstm_cell/MatMul_grad/MatMulMatMulCgradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
ь
2gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul_1MatMulrnn/basic_lstm_cell/concatCgradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
Њ
:gradients/rnn/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul3^gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul_1
Д
Bgradients/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul;^gradients/rnn/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Б
Dgradients/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul_1;^gradients/rnn/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul_1*
_output_shapes

:P
Љ

gradients/AddN_15AddNHgradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/control_dependency_1Hgradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/control_dependency_1Hgradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/control_dependency_1Hgradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/control_dependency_1Hgradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/control_dependency_1Hgradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependency_1Egradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/BiasAddGrad*
N*
T0*
_output_shapes
:P


gradients/AddN_16AddNGgradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependency_1Dgradients/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul_1*
N*
_output_shapes

:P
}
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
_class
loc:@dense/bias*
dtype0*
	container *
shape: *
shared_name *
_output_shapes
: 
­
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 
i
beta1_power/readIdentitybeta1_power*
_class
loc:@dense/bias*
T0*
_output_shapes
: 
}
beta2_power/initial_valueConst*
valueB
 *wО?*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
shape: *
_output_shapes
: 
­
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
use_locking(*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
i
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
С
Arnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"   P   *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes
:
Ћ
7rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 

1rnn/basic_lstm_cell/kernel/Adam/Initializer/zerosFillArnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensor7rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
Т
rnn/basic_lstm_cell/kernel/Adam
VariableV2*
shape
:P*
shared_name *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
	container *
_output_shapes

:P

&rnn/basic_lstm_cell/kernel/Adam/AssignAssignrnn/basic_lstm_cell/kernel/Adam1rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:P
Љ
$rnn/basic_lstm_cell/kernel/Adam/readIdentityrnn/basic_lstm_cell/kernel/Adam*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
У
Crnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   P   *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:
­
9rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Ѕ
3rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillCrnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensor9rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
Ф
!rnn/basic_lstm_cell/kernel/Adam_1
VariableV2*
	container *
shape
:P*
shared_name *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes

:P

(rnn/basic_lstm_cell/kernel/Adam_1/AssignAssign!rnn/basic_lstm_cell/kernel/Adam_13rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
use_locking(*
_output_shapes

:P
­
&rnn/basic_lstm_cell/kernel/Adam_1/readIdentity!rnn/basic_lstm_cell/kernel/Adam_1*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
Љ
/rnn/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
valueBP*    *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
:P
Ж
rnn/basic_lstm_cell/bias/Adam
VariableV2*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
	container *
shape:P*
shared_name *
_output_shapes
:P
љ
$rnn/basic_lstm_cell/bias/Adam/AssignAssignrnn/basic_lstm_cell/bias/Adam/rnn/basic_lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:P

"rnn/basic_lstm_cell/bias/Adam/readIdentityrnn/basic_lstm_cell/bias/Adam*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes
:P
Ћ
1rnn/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
valueBP*    *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
:P
И
rnn/basic_lstm_cell/bias/Adam_1
VariableV2*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
	container *
shape:P*
shared_name *
_output_shapes
:P
џ
&rnn/basic_lstm_cell/bias/Adam_1/AssignAssignrnn/basic_lstm_cell/bias/Adam_11rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
use_locking(*
_output_shapes
:P
Ѓ
$rnn/basic_lstm_cell/bias/Adam_1/readIdentityrnn/basic_lstm_cell/bias/Adam_1*+
_class!
loc:@rnn/basic_lstm_cell/bias*
T0*
_output_shapes
:P

#dense/kernel/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/kernel*
dtype0*
_output_shapes

:
І
dense/kernel/Adam
VariableV2*
shape
:*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container *
_output_shapes

:
Э
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
T0*
_class
loc:@dense/kernel*
validate_shape(*
use_locking(*
_output_shapes

:

dense/kernel/Adam/readIdentitydense/kernel/Adam*
T0*
_class
loc:@dense/kernel*
_output_shapes

:

%dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/kernel*
dtype0*
_output_shapes

:
Ј
dense/kernel/Adam_1
VariableV2*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container *
shape
:*
_output_shapes

:
г
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:

dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
_class
loc:@dense/kernel*
T0*
_output_shapes

:

!dense/bias/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:

dense/bias/Adam
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_class
loc:@dense/bias*
_output_shapes
:
С
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
u
dense/bias/Adam/readIdentitydense/bias/Adam*
_class
loc:@dense/bias*
T0*
_output_shapes
:

#dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:

dense/bias/Adam_1
VariableV2*
_class
loc:@dense/bias*
dtype0*
	container *
shape:*
shared_name *
_output_shapes
:
Ч
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
y
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
T0*
_class
loc:@dense/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 

0Adam/update_rnn/basic_lstm_cell/kernel/ApplyAdam	ApplyAdamrnn/basic_lstm_cell/kernelrnn/basic_lstm_cell/kernel/Adam!rnn/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_16*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
use_nesterov( *
use_locking( *
_output_shapes

:P
џ
.Adam/update_rnn/basic_lstm_cell/bias/ApplyAdam	ApplyAdamrnn/basic_lstm_cell/biasrnn/basic_lstm_cell/bias/Adamrnn/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_15*
use_nesterov( *
use_locking( *
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes
:P
ь
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
use_nesterov( *
_output_shapes

:
п
 Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/bias*
use_nesterov( *
_output_shapes
:

Adam/mulMulbeta1_power/read
Adam/beta1!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam/^Adam/update_rnn/basic_lstm_cell/bias/ApplyAdam1^Adam/update_rnn/basic_lstm_cell/kernel/ApplyAdam*
T0*
_class
loc:@dense/bias*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_class
loc:@dense/bias*
validate_shape(*
use_locking( *
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam/^Adam/update_rnn/basic_lstm_cell/bias/ApplyAdam1^Adam/update_rnn/basic_lstm_cell/kernel/ApplyAdam*
T0*
_class
loc:@dense/bias*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_class
loc:@dense/bias*
validate_shape(*
use_locking( *
T0*
_output_shapes
: 
н
Adam/updateNoOp^Adam/Assign^Adam/Assign_1!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam/^Adam/update_rnn/basic_lstm_cell/bias/ApplyAdam1^Adam/update_rnn/basic_lstm_cell/kernel/ApplyAdam
z

Adam/valueConst^Adam/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
~
Adam	AssignAddglobal_step
Adam/value*
use_locking( *
T0	*
_class
loc:@global_step*
_output_shapes
: 
^
subSubIteratorGetNext:1dense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
G
SquareSquaresub*
T0*'
_output_shapes
:џџџџџџџџџ
І
/root_mean_squared_error/total/Initializer/zerosConst*
dtype0*
valueB
 *    *0
_class&
$"loc:@root_mean_squared_error/total*
_output_shapes
: 
Г
root_mean_squared_error/total
VariableV2*
shape: *
shared_name *0
_class&
$"loc:@root_mean_squared_error/total*
dtype0*
	container *
_output_shapes
: 
њ
$root_mean_squared_error/total/AssignAssignroot_mean_squared_error/total/root_mean_squared_error/total/Initializer/zeros*
T0*0
_class&
$"loc:@root_mean_squared_error/total*
validate_shape(*
use_locking(*
_output_shapes
: 
 
"root_mean_squared_error/total/readIdentityroot_mean_squared_error/total*0
_class&
$"loc:@root_mean_squared_error/total*
T0*
_output_shapes
: 
І
/root_mean_squared_error/count/Initializer/zerosConst*
valueB
 *    *0
_class&
$"loc:@root_mean_squared_error/count*
dtype0*
_output_shapes
: 
Г
root_mean_squared_error/count
VariableV2*
shape: *
shared_name *0
_class&
$"loc:@root_mean_squared_error/count*
dtype0*
	container *
_output_shapes
: 
њ
$root_mean_squared_error/count/AssignAssignroot_mean_squared_error/count/root_mean_squared_error/count/Initializer/zeros*
T0*0
_class&
$"loc:@root_mean_squared_error/count*
validate_shape(*
use_locking(*
_output_shapes
: 
 
"root_mean_squared_error/count/readIdentityroot_mean_squared_error/count*
T0*0
_class&
$"loc:@root_mean_squared_error/count*
_output_shapes
: 
]
root_mean_squared_error/SizeSizeSquare*
T0*
out_type0*
_output_shapes
: 
w
!root_mean_squared_error/ToFloat_1Castroot_mean_squared_error/Size*

DstT0*

SrcT0*
_output_shapes
: 
n
root_mean_squared_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

root_mean_squared_error/SumSumSquareroot_mean_squared_error/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
а
!root_mean_squared_error/AssignAdd	AssignAddroot_mean_squared_error/totalroot_mean_squared_error/Sum*
T0*0
_class&
$"loc:@root_mean_squared_error/total*
use_locking( *
_output_shapes
: 
с
#root_mean_squared_error/AssignAdd_1	AssignAddroot_mean_squared_error/count!root_mean_squared_error/ToFloat_1^Square*
use_locking( *
T0*0
_class&
$"loc:@root_mean_squared_error/count*
_output_shapes
: 

root_mean_squared_error/truedivRealDiv"root_mean_squared_error/total/read"root_mean_squared_error/count/read*
T0*
_output_shapes
: 
g
"root_mean_squared_error/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 

root_mean_squared_error/GreaterGreater"root_mean_squared_error/count/read"root_mean_squared_error/zeros_like*
T0*
_output_shapes
: 
Ў
root_mean_squared_error/valueSelectroot_mean_squared_error/Greaterroot_mean_squared_error/truediv"root_mean_squared_error/zeros_like*
T0*
_output_shapes
: 

!root_mean_squared_error/truediv_1RealDiv!root_mean_squared_error/AssignAdd#root_mean_squared_error/AssignAdd_1*
T0*
_output_shapes
: 
i
$root_mean_squared_error/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 

!root_mean_squared_error/Greater_1Greater#root_mean_squared_error/AssignAdd_1$root_mean_squared_error/zeros_like_1*
T0*
_output_shapes
: 
И
!root_mean_squared_error/update_opSelect!root_mean_squared_error/Greater_1!root_mean_squared_error/truediv_1$root_mean_squared_error/zeros_like_1*
T0*
_output_shapes
: 
L
SqrtSqrtroot_mean_squared_error/value*
T0*
_output_shapes
: 
R
Sqrt_1Sqrt!root_mean_squared_error/update_op*
T0*
_output_shapes
: 
`
sub_1Subdense/BiasAddIteratorGetNext:1*
T0*'
_output_shapes
:џџџџџџџџџ
C
AbsAbssub_1*
T0*'
_output_shapes
:џџџџџџџџџ

+mean_absolute_error/total/Initializer/zerosConst*
valueB
 *    *,
_class"
 loc:@mean_absolute_error/total*
dtype0*
_output_shapes
: 
Ћ
mean_absolute_error/total
VariableV2*,
_class"
 loc:@mean_absolute_error/total*
dtype0*
	container *
shape: *
shared_name *
_output_shapes
: 
ъ
 mean_absolute_error/total/AssignAssignmean_absolute_error/total+mean_absolute_error/total/Initializer/zeros*
T0*,
_class"
 loc:@mean_absolute_error/total*
validate_shape(*
use_locking(*
_output_shapes
: 

mean_absolute_error/total/readIdentitymean_absolute_error/total*,
_class"
 loc:@mean_absolute_error/total*
T0*
_output_shapes
: 

+mean_absolute_error/count/Initializer/zerosConst*
valueB
 *    *,
_class"
 loc:@mean_absolute_error/count*
dtype0*
_output_shapes
: 
Ћ
mean_absolute_error/count
VariableV2*
shared_name *,
_class"
 loc:@mean_absolute_error/count*
dtype0*
	container *
shape: *
_output_shapes
: 
ъ
 mean_absolute_error/count/AssignAssignmean_absolute_error/count+mean_absolute_error/count/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@mean_absolute_error/count*
validate_shape(*
_output_shapes
: 

mean_absolute_error/count/readIdentitymean_absolute_error/count*
T0*,
_class"
 loc:@mean_absolute_error/count*
_output_shapes
: 
V
mean_absolute_error/SizeSizeAbs*
T0*
out_type0*
_output_shapes
: 
o
mean_absolute_error/ToFloat_1Castmean_absolute_error/Size*

SrcT0*

DstT0*
_output_shapes
: 
j
mean_absolute_error/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
|
mean_absolute_error/SumSumAbsmean_absolute_error/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Р
mean_absolute_error/AssignAdd	AssignAddmean_absolute_error/totalmean_absolute_error/Sum*
use_locking( *
T0*,
_class"
 loc:@mean_absolute_error/total*
_output_shapes
: 
Ю
mean_absolute_error/AssignAdd_1	AssignAddmean_absolute_error/countmean_absolute_error/ToFloat_1^Abs*,
_class"
 loc:@mean_absolute_error/count*
use_locking( *
T0*
_output_shapes
: 

mean_absolute_error/truedivRealDivmean_absolute_error/total/readmean_absolute_error/count/read*
T0*
_output_shapes
: 
c
mean_absolute_error/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 

mean_absolute_error/GreaterGreatermean_absolute_error/count/readmean_absolute_error/zeros_like*
T0*
_output_shapes
: 

mean_absolute_error/valueSelectmean_absolute_error/Greatermean_absolute_error/truedivmean_absolute_error/zeros_like*
T0*
_output_shapes
: 

mean_absolute_error/truediv_1RealDivmean_absolute_error/AssignAddmean_absolute_error/AssignAdd_1*
T0*
_output_shapes
: 
e
 mean_absolute_error/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 

mean_absolute_error/Greater_1Greatermean_absolute_error/AssignAdd_1 mean_absolute_error/zeros_like_1*
T0*
_output_shapes
: 
Ј
mean_absolute_error/update_opSelectmean_absolute_error/Greater_1mean_absolute_error/truediv_1 mean_absolute_error/zeros_like_1*
T0*
_output_shapes
: 

mean/total/Initializer/zerosConst*
valueB
 *    *
_class
loc:@mean/total*
dtype0*
_output_shapes
: 


mean/total
VariableV2*
dtype0*
	container *
shape: *
shared_name *
_class
loc:@mean/total*
_output_shapes
: 
Ў
mean/total/AssignAssign
mean/totalmean/total/Initializer/zeros*
use_locking(*
T0*
_class
loc:@mean/total*
validate_shape(*
_output_shapes
: 
g
mean/total/readIdentity
mean/total*
T0*
_class
loc:@mean/total*
_output_shapes
: 

mean/count/Initializer/zerosConst*
valueB
 *    *
_class
loc:@mean/count*
dtype0*
_output_shapes
: 


mean/count
VariableV2*
_class
loc:@mean/count*
dtype0*
	container *
shape: *
shared_name *
_output_shapes
: 
Ў
mean/count/AssignAssign
mean/countmean/count/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@mean/count*
_output_shapes
: 
g
mean/count/readIdentity
mean/count*
T0*
_class
loc:@mean/count*
_output_shapes
: 
K
	mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Q
mean/ToFloat_1Cast	mean/Size*

SrcT0*

DstT0*
_output_shapes
: 
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
s
mean/SumSummean_squared_error/value
mean/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 

mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
T0*
_class
loc:@mean/total*
use_locking( *
_output_shapes
: 
Ї
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1^mean_squared_error/value*
use_locking( *
T0*
_class
loc:@mean/count*
_output_shapes
: 
Z
mean/truedivRealDivmean/total/readmean/count/read*
T0*
_output_shapes
: 
T
mean/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
mean/GreaterGreatermean/count/readmean/zeros_like*
T0*
_output_shapes
: 
b

mean/valueSelectmean/Greatermean/truedivmean/zeros_like*
T0*
_output_shapes
: 
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
T0*
_output_shapes
: 
V
mean/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
_
mean/Greater_1Greatermean/AssignAdd_1mean/zeros_like_1*
T0*
_output_shapes
: 
l
mean/update_opSelectmean/Greater_1mean/truediv_1mean/zeros_like_1*
T0*
_output_shapes
: 
L

group_depsNoOp^Sqrt_1^mean/update_op^mean_absolute_error/update_op
{
eval_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 

	eval_step
VariableV2*
_class
loc:@eval_step*
dtype0	*
	container *
shape: *
shared_name *
_output_shapes
: 
Њ
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@eval_step*
validate_shape(*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
_class
loc:@eval_step*
T0	*
_output_shapes
: 
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 

	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
use_locking(*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
U
readIdentity	eval_step
^AssignAdd^group_deps*
T0	*
_output_shapes
: 
;
IdentityIdentityread*
T0	*
_output_shapes
: 
Ы
initNoOp^beta1_power/Assign^beta2_power/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^global_step/Assign%^rnn/basic_lstm_cell/bias/Adam/Assign'^rnn/basic_lstm_cell/bias/Adam_1/Assign ^rnn/basic_lstm_cell/bias/Assign'^rnn/basic_lstm_cell/kernel/Adam/Assign)^rnn/basic_lstm_cell/kernel/Adam_1/Assign"^rnn/basic_lstm_cell/kernel/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
П
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedrnn/basic_lstm_cell/kernel*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Л
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedrnn/basic_lstm_cell/bias*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
: 
Ѓ
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitialized
dense/bias*
dtype0*
_class
loc:@dense/bias*
_output_shapes
: 
 
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedbeta1_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
 
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializedbeta2_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ф
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializedrnn/basic_lstm_cell/kernel/Adam*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Ц
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized!rnn/basic_lstm_cell/kernel/Adam_1*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Р
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedrnn/basic_lstm_cell/bias/Adam*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
: 
У
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedrnn/basic_lstm_cell/bias/Adam_1*
dtype0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes
: 
Љ
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializeddense/kernel/Adam*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ћ
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializeddense/kernel/Adam_1*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ѕ
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializeddense/bias/Adam*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ї
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializeddense/bias/Adam_1*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ц
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedroot_mean_squared_error/total*0
_class&
$"loc:@root_mean_squared_error/total*
dtype0*
_output_shapes
: 
Ц
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedroot_mean_squared_error/count*0
_class&
$"loc:@root_mean_squared_error/count*
dtype0*
_output_shapes
: 
О
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedmean_absolute_error/total*,
_class"
 loc:@mean_absolute_error/total*
dtype0*
_output_shapes
: 
О
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedmean_absolute_error/count*,
_class"
 loc:@mean_absolute_error/count*
dtype0*
_output_shapes
: 
 
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized
mean/total*
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
 
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 

7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
Я

$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_21"/device:CPU:0*

axis *
N*
T0
*
_output_shapes
:

)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
л
$report_uninitialized_variables/ConstConst"/device:CPU:0*ѓ
valueщBцBglobal_stepBrnn/basic_lstm_cell/kernelBrnn/basic_lstm_cell/biasBdense/kernelB
dense/biasBbeta1_powerBbeta2_powerBrnn/basic_lstm_cell/kernel/AdamB!rnn/basic_lstm_cell/kernel/Adam_1Brnn/basic_lstm_cell/bias/AdamBrnn/basic_lstm_cell/bias/Adam_1Bdense/kernel/AdamBdense/kernel/Adam_1Bdense/bias/AdamBdense/bias/Adam_1Broot_mean_squared_error/totalBroot_mean_squared_error/countBmean_absolute_error/totalBmean_absolute_error/countB
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:

1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
dtype0*
valueB: *
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
ш
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
_output_shapes
:

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 

3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
dtype0*
valueB: *
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
№
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
_output_shapes
: 

3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
№
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
О
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*

axis *
N*
T0*
_output_shapes
:

7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
ї
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*
T0*
N*

Tidx0*
_output_shapes
:
к
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
Tshape0*
T0*
_output_shapes
:

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
dtype0*
valueB:
џџџџџџџџџ*
_output_shapes
:
ъ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
В
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:џџџџџџџџџ
Х
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ

9report_uninitialized_variables/boolean_mask/GatherV2/axisConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
Х
4report_uninitialized_variables/boolean_mask/GatherV2GatherV23report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze9report_uninitialized_variables/boolean_mask/GatherV2/axis"/device:CPU:0*
Tparams0*
Taxis0*
Tindices0	*#
_output_shapes
:џџџџџџџџџ
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
О
concatConcatV24report_uninitialized_variables/boolean_mask/GatherV2$report_uninitialized_resources/Constconcat/axis*
N*

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ
Ё
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
С
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedrnn/basic_lstm_cell/kernel*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Н
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedrnn/basic_lstm_cell/bias*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
: 
Ѕ
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ё
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ђ
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializedbeta1_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ђ
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializedbeta2_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ц
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializedrnn/basic_lstm_cell/kernel/Adam*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Ш
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitialized!rnn/basic_lstm_cell/kernel/Adam_1*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Т
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializedrnn/basic_lstm_cell/bias/Adam*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
: 
Х
9report_uninitialized_variables_1/IsVariableInitialized_10IsVariableInitializedrnn/basic_lstm_cell/bias/Adam_1*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
: 
Ћ
9report_uninitialized_variables_1/IsVariableInitialized_11IsVariableInitializeddense/kernel/Adam*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
­
9report_uninitialized_variables_1/IsVariableInitialized_12IsVariableInitializeddense/kernel/Adam_1*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ї
9report_uninitialized_variables_1/IsVariableInitialized_13IsVariableInitializeddense/bias/Adam*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Љ
9report_uninitialized_variables_1/IsVariableInitialized_14IsVariableInitializeddense/bias/Adam_1*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
р
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_99report_uninitialized_variables_1/IsVariableInitialized_109report_uninitialized_variables_1/IsVariableInitialized_119report_uninitialized_variables_1/IsVariableInitialized_129report_uninitialized_variables_1/IsVariableInitialized_139report_uninitialized_variables_1/IsVariableInitialized_14"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:

+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:
Ц
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*м
valueвBЯBglobal_stepBrnn/basic_lstm_cell/kernelBrnn/basic_lstm_cell/biasBdense/kernelB
dense/biasBbeta1_powerBbeta2_powerBrnn/basic_lstm_cell/kernel/AdamB!rnn/basic_lstm_cell/kernel/Adam_1Brnn/basic_lstm_cell/bias/AdamBrnn/basic_lstm_cell/bias/Adam_1Bdense/kernel/AdamBdense/kernel/Adam_1Bdense/bias/AdamBdense/bias/Adam_1*
dtype0*
_output_shapes
:

3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
ђ
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:

Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
dtype0*
valueB:*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
T0*
Index0*
_output_shapes
: 
Т
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
N*

Tidx0*
T0*
_output_shapes
:
р
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:

=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
№
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
Ж
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:џџџџџџџџџ
Щ
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ

;report_uninitialized_variables_1/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Э
6report_uninitialized_variables_1/boolean_mask/GatherV2GatherV25report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze;report_uninitialized_variables_1/boolean_mask/GatherV2/axis"/device:CPU:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ
н
init_2NoOp^eval_step/Assign^mean/count/Assign^mean/total/Assign!^mean_absolute_error/count/Assign!^mean_absolute_error/total/Assign%^root_mean_squared_error/count/Assign%^root_mean_squared_error/total/Assign

init_all_tablesNoOp

init_3NoOp
8
group_deps_2NoOp^init_2^init_3^init_all_tables
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_2f099ea2be324ff7a3a206e8b7b3e60a/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
И
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*м
valueвBЯBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bglobal_stepBrnn/basic_lstm_cell/biasBrnn/basic_lstm_cell/bias/AdamBrnn/basic_lstm_cell/bias/Adam_1Brnn/basic_lstm_cell/kernelBrnn/basic_lstm_cell/kernel/AdamB!rnn/basic_lstm_cell/kernel/Adam_1*
_output_shapes
:

save/SaveV2/shape_and_slicesConst"/device:CPU:0*1
value(B&B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
и
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1global_steprnn/basic_lstm_cell/biasrnn/basic_lstm_cell/bias/Adamrnn/basic_lstm_cell/bias/Adam_1rnn/basic_lstm_cell/kernelrnn/basic_lstm_cell/kernel/Adam!rnn/basic_lstm_cell/kernel/Adam_1"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
Л
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*м
valueвBЯBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bglobal_stepBrnn/basic_lstm_cell/biasBrnn/basic_lstm_cell/bias/AdamBrnn/basic_lstm_cell/bias/Adam_1Brnn/basic_lstm_cell/kernelBrnn/basic_lstm_cell/kernel/AdamB!rnn/basic_lstm_cell/kernel/Adam_1*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*1
value(B&B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
х
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*P
_output_shapes>
<:::::::::::::::

save/AssignAssignbeta1_powersave/RestoreV2*
_class
loc:@dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 

save/Assign_1Assignbeta2_powersave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
Ђ
save/Assign_2Assign
dense/biassave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
Ї
save/Assign_3Assigndense/bias/Adamsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
Љ
save/Assign_4Assigndense/bias/Adam_1save/RestoreV2:4*
T0*
_class
loc:@dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:
Њ
save/Assign_5Assigndense/kernelsave/RestoreV2:5*
_class
loc:@dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:
Џ
save/Assign_6Assigndense/kernel/Adamsave/RestoreV2:6*
validate_shape(*
use_locking(*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
Б
save/Assign_7Assigndense/kernel/Adam_1save/RestoreV2:7*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
 
save/Assign_8Assignglobal_stepsave/RestoreV2:8*
T0	*
_class
loc:@global_step*
validate_shape(*
use_locking(*
_output_shapes
: 
О
save/Assign_9Assignrnn/basic_lstm_cell/biassave/RestoreV2:9*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:P
Х
save/Assign_10Assignrnn/basic_lstm_cell/bias/Adamsave/RestoreV2:10*
validate_shape(*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes
:P
Ч
save/Assign_11Assignrnn/basic_lstm_cell/bias/Adam_1save/RestoreV2:11*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:P
Ш
save/Assign_12Assignrnn/basic_lstm_cell/kernelsave/RestoreV2:12*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:P
Э
save/Assign_13Assignrnn/basic_lstm_cell/kernel/Adamsave/RestoreV2:13*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:P
Я
save/Assign_14Assign!rnn/basic_lstm_cell/kernel/Adam_1save/RestoreV2:14*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
use_locking(*
_output_shapes

:P

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shardХ'
Ѓ
1
_make_dataset_IBZ7uUdCWTo
repeatdatasetb
%TextLineDataset/MatchingFiles/patternConst*%
valueB Bdata/seq01.train.csv*
dtype0^
TextLineDataset/MatchingFilesMatchingFiles.TextLineDataset/MatchingFiles/pattern:output:0I
 TextLineDataset/compression_typeConst*
valueB B *
dtype0G
TextLineDataset/buffer_sizeConst*
valueB		 R*
dtype0	
TextLineDatasetTextLineDataset)TextLineDataset/MatchingFiles:filenames:0)TextLineDataset/compression_type:output:0$TextLineDataset/buffer_size:output:0;
SkipDataset/countConst*
value	B	 R *
dtype0	~
SkipDatasetSkipDatasetTextLineDataset:handle:0SkipDataset/count:output:0*
output_types
2*
output_shapes
: A
BatchDataset/batch_sizeConst*
value	B	 Rd*
dtype0	
BatchDatasetBatchDatasetSkipDataset:handle:0 BatchDataset/batch_size:output:0*
output_types
2*"
output_shapes
:џџџџџџџџџO
%ParallelMapDataset/num_parallel_callsConst*
value	B :*
dtype0і
ParallelMapDatasetParallelMapDatasetBatchDataset:handle:0.ParallelMapDataset/num_parallel_calls:output:0*

Targuments
 *9
output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ* 
fR
tf_map_func_1xNaTDyIc3E*
output_types
2?
RepeatDataset/count_1Const*
value	B	 R*
dtype0	Ў
RepeatDatasetRepeatDatasetParallelMapDataset:handle:0RepeatDataset/count_1:output:0*
output_types
2*9
output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ"'
repeatdatasetRepeatDataset:handle:0

t
tf_map_func_1xNaTDyIc3E
arg0

concat
concat_125A wrapper for Defun that facilitates shape inference.A
ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0L

ExpandDims
ExpandDimsarg0ExpandDims/dim:output:0*
T0*

Tdim0L
DecodeCSV/record_defaults_0Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_1Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_2Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_3Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_4Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_5Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_6Const*
dtype0*
valueB*    L
DecodeCSV/record_defaults_7Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_8Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_9Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_10Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_11Const*
dtype0*
valueB*    M
DecodeCSV/record_defaults_12Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_13Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_14Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_15Const*
dtype0*
valueB*    M
DecodeCSV/record_defaults_16Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_17Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_18Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_19Const*
valueB*    *
dtype0
	DecodeCSV	DecodeCSVExpandDims:output:0$DecodeCSV/record_defaults_0:output:0$DecodeCSV/record_defaults_1:output:0$DecodeCSV/record_defaults_2:output:0$DecodeCSV/record_defaults_3:output:0$DecodeCSV/record_defaults_4:output:0$DecodeCSV/record_defaults_5:output:0$DecodeCSV/record_defaults_6:output:0$DecodeCSV/record_defaults_7:output:0$DecodeCSV/record_defaults_8:output:0$DecodeCSV/record_defaults_9:output:0%DecodeCSV/record_defaults_10:output:0%DecodeCSV/record_defaults_11:output:0%DecodeCSV/record_defaults_12:output:0%DecodeCSV/record_defaults_13:output:0%DecodeCSV/record_defaults_14:output:0%DecodeCSV/record_defaults_15:output:0%DecodeCSV/record_defaults_16:output:0%DecodeCSV/record_defaults_17:output:0%DecodeCSV/record_defaults_18:output:0%DecodeCSV/record_defaults_19:output:0*$
OUT_TYPE
2*
field_delim,*
na_value *
use_quote_delim(5
concat/axisConst*
dtype0*
value	B :
concat_0ConcatV2DecodeCSV:output:0DecodeCSV:output:1DecodeCSV:output:2DecodeCSV:output:3DecodeCSV:output:4DecodeCSV:output:5DecodeCSV:output:6DecodeCSV:output:7DecodeCSV:output:8DecodeCSV:output:9DecodeCSV:output:10DecodeCSV:output:11DecodeCSV:output:12DecodeCSV:output:13DecodeCSV:output:14DecodeCSV:output:15concat/axis:output:0*
N*

Tidx0*
T07
concat_1/axisConst*
dtype0*
value	B : 

concat_1_0ConcatV2DecodeCSV:output:16DecodeCSV:output:17DecodeCSV:output:18DecodeCSV:output:19concat_1/axis:output:0*

Tidx0*
T0*
N"
concat_1concat_1_0:output:0"
concatconcat_0:output:0"ЁќђЪа     њћ*	!
P%ЫжAJНЁ
*џ)
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
ю
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype

IteratorGetNext
iterator

components2output_types"
output_types
list(type)(0" 
output_shapeslist(shape)(0
C
IteratorToStringHandle
resource_handle
string_handle


LogicalNot
x

y

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
+
MatchingFiles
pattern
	filenames
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
Џ
OneShotIterator

handle"
dataset_factoryfunc"
output_types
list(type)(0" 
output_shapeslist(shape)(0"
	containerstring "
shared_namestring 
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
E
Where

input"T	
index	"%
Ttype0
:
2	
*1.8.02
b'unknown'и

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

global_step
VariableV2*
shape: *
shared_name *
_class
loc:@global_step*
dtype0	*
	container *
_output_shapes
: 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
y
MatchingFiles/patternConst"/device:CPU:0*%
valueB Bdata/seq01.train.csv*
dtype0*
_output_shapes
: 
i
MatchingFilesMatchingFilesMatchingFiles/pattern"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ
`
compression_typeConst"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
^
buffer_sizeConst"/device:CPU:0*
valueB		 R*
dtype0	*
_output_shapes
: 
V
countConst"/device:CPU:0*
dtype0	*
value	B	 R *
_output_shapes
: 
[

batch_sizeConst"/device:CPU:0*
value	B	 Rd*
dtype0	*
_output_shapes
: 
c
num_parallel_callsConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
X
count_1Const"/device:CPU:0*
value	B	 R*
dtype0	*
_output_shapes
: 
ђ
OneShotIteratorOneShotIterator"/device:CPU:0*
shared_name *9
output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*
	container *0
dataset_factoryR
_make_dataset_IBZ7uUdCWTo*
output_types
2*
_output_shapes
: 
h
IteratorToStringHandleIteratorToStringHandleOneShotIterator"/device:CPU:0*
_output_shapes
: 
б
IteratorGetNextIteratorGetNextOneShotIterator"/device:CPU:0*
output_types
2*9
output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
G
ConstConst*
value	B :*
dtype0*
_output_shapes
: 
Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 

splitSplitsplit/split_dimIteratorGetNext*
T0*
	num_split*Ц
_output_shapesГ
А:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
N
	rnn/ShapeShapesplit*
T0*
out_type0*
_output_shapes
:
a
rnn/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
c
rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
c
rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
k
)rnn/BasicLSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ђ
%rnn/BasicLSTMCellZeroState/ExpandDims
ExpandDimsrnn/strided_slice)rnn/BasicLSTMCellZeroState/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:
j
 rnn/BasicLSTMCellZeroState/ConstConst*
dtype0*
valueB:*
_output_shapes
:
h
&rnn/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
и
!rnn/BasicLSTMCellZeroState/concatConcatV2%rnn/BasicLSTMCellZeroState/ExpandDims rnn/BasicLSTMCellZeroState/Const&rnn/BasicLSTMCellZeroState/concat/axis*
N*

Tidx0*
T0*
_output_shapes
:
k
&rnn/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
З
 rnn/BasicLSTMCellZeroState/zerosFill!rnn/BasicLSTMCellZeroState/concat&rnn/BasicLSTMCellZeroState/zeros/Const*

index_type0*
T0*'
_output_shapes
:џџџџџџџџџ
m
+rnn/BasicLSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
І
'rnn/BasicLSTMCellZeroState/ExpandDims_1
ExpandDimsrnn/strided_slice+rnn/BasicLSTMCellZeroState/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
m
+rnn/BasicLSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
І
'rnn/BasicLSTMCellZeroState/ExpandDims_2
ExpandDimsrnn/strided_slice+rnn/BasicLSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
j
(rnn/BasicLSTMCellZeroState/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
р
#rnn/BasicLSTMCellZeroState/concat_1ConcatV2'rnn/BasicLSTMCellZeroState/ExpandDims_2"rnn/BasicLSTMCellZeroState/Const_2(rnn/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*

Tidx0*
_output_shapes
:
m
(rnn/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
"rnn/BasicLSTMCellZeroState/zeros_1Fill#rnn/BasicLSTMCellZeroState/concat_1(rnn/BasicLSTMCellZeroState/zeros_1/Const*

index_type0*
T0*'
_output_shapes
:џџџџџџџџџ
m
+rnn/BasicLSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
І
'rnn/BasicLSTMCellZeroState/ExpandDims_3
ExpandDimsrnn/strided_slice+rnn/BasicLSTMCellZeroState/ExpandDims_3/dim*
T0*

Tdim0*
_output_shapes
:
l
"rnn/BasicLSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:
Л
;rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"   P   *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:
­
9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *<yО*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
­
9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *<y>*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 

Crnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform;rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
seed2.*
seedвЎК	*
_output_shapes

:P

9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSub9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/max9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes
: 

9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulCrnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P

5rnn/basic_lstm_cell/kernel/Initializer/random_uniformAdd9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mul9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
Н
rnn/basic_lstm_cell/kernel
VariableV2*
shape
:P*
shared_name *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
	container *
_output_shapes

:P
џ
!rnn/basic_lstm_cell/kernel/AssignAssignrnn/basic_lstm_cell/kernel5rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
use_locking(*
_output_shapes

:P
p
rnn/basic_lstm_cell/kernel/readIdentityrnn/basic_lstm_cell/kernel*
T0*
_output_shapes

:P
Є
*rnn/basic_lstm_cell/bias/Initializer/zerosConst*
valueBP*    *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
:P
Б
rnn/basic_lstm_cell/bias
VariableV2*
shared_name *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
	container *
shape:P*
_output_shapes
:P
ъ
rnn/basic_lstm_cell/bias/AssignAssignrnn/basic_lstm_cell/bias*rnn/basic_lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:P
h
rnn/basic_lstm_cell/bias/readIdentityrnn/basic_lstm_cell/bias*
T0*
_output_shapes
:P
[
rnn/basic_lstm_cell/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
a
rnn/basic_lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Й
rnn/basic_lstm_cell/concatConcatV2split"rnn/BasicLSTMCellZeroState/zeros_1rnn/basic_lstm_cell/concat/axis*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџ
Й
rnn/basic_lstm_cell/MatMulMatMulrnn/basic_lstm_cell/concatrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Њ
rnn/basic_lstm_cell/BiasAddBiasAddrnn/basic_lstm_cell/MatMulrnn/basic_lstm_cell/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџP
]
rnn/basic_lstm_cell/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
ж
rnn/basic_lstm_cell/splitSplitrnn/basic_lstm_cell/Constrnn/basic_lstm_cell/BiasAdd*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
`
rnn/basic_lstm_cell/Const_2Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 

rnn/basic_lstm_cell/AddAddrnn/basic_lstm_cell/split:2rnn/basic_lstm_cell/Const_2*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/SigmoidSigmoidrnn/basic_lstm_cell/Add*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/MulMul rnn/BasicLSTMCellZeroState/zerosrnn/basic_lstm_cell/Sigmoid*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Sigmoid_1Sigmoidrnn/basic_lstm_cell/split*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/TanhTanhrnn/basic_lstm_cell/split:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_1rnn/basic_lstm_cell/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_1Addrnn/basic_lstm_cell/Mulrnn/basic_lstm_cell/Mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/Tanh_1Tanhrnn/basic_lstm_cell/Add_1*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_2Sigmoidrnn/basic_lstm_cell/split:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_2Mulrnn/basic_lstm_cell/Tanh_1rnn/basic_lstm_cell/Sigmoid_2*
T0*'
_output_shapes
:џџџџџџџџџ
]
rnn/basic_lstm_cell/Const_3Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
rnn/basic_lstm_cell/concat_1ConcatV2split:1rnn/basic_lstm_cell/Mul_2!rnn/basic_lstm_cell/concat_1/axis*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_1MatMulrnn/basic_lstm_cell/concat_1rnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_1BiasAddrnn/basic_lstm_cell/MatMul_1rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
]
rnn/basic_lstm_cell/Const_4Const*
value	B :*
dtype0*
_output_shapes
: 
м
rnn/basic_lstm_cell/split_1Splitrnn/basic_lstm_cell/Const_3rnn/basic_lstm_cell/BiasAdd_1*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
`
rnn/basic_lstm_cell/Const_5Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_2Addrnn/basic_lstm_cell/split_1:2rnn/basic_lstm_cell/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Sigmoid_3Sigmoidrnn/basic_lstm_cell/Add_2*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_3Mulrnn/basic_lstm_cell/Add_1rnn/basic_lstm_cell/Sigmoid_3*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_4Sigmoidrnn/basic_lstm_cell/split_1*
T0*'
_output_shapes
:џџџџџџџџџ
s
rnn/basic_lstm_cell/Tanh_2Tanhrnn/basic_lstm_cell/split_1:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_4Mulrnn/basic_lstm_cell/Sigmoid_4rnn/basic_lstm_cell/Tanh_2*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_3Addrnn/basic_lstm_cell/Mul_3rnn/basic_lstm_cell/Mul_4*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/Tanh_3Tanhrnn/basic_lstm_cell/Add_3*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_5Sigmoidrnn/basic_lstm_cell/split_1:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_5Mulrnn/basic_lstm_cell/Tanh_3rnn/basic_lstm_cell/Sigmoid_5*
T0*'
_output_shapes
:џџџџџџџџџ
]
rnn/basic_lstm_cell/Const_6Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_2/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
rnn/basic_lstm_cell/concat_2ConcatV2split:2rnn/basic_lstm_cell/Mul_5!rnn/basic_lstm_cell/concat_2/axis*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_2MatMulrnn/basic_lstm_cell/concat_2rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_2BiasAddrnn/basic_lstm_cell/MatMul_2rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
]
rnn/basic_lstm_cell/Const_7Const*
value	B :*
dtype0*
_output_shapes
: 
м
rnn/basic_lstm_cell/split_2Splitrnn/basic_lstm_cell/Const_6rnn/basic_lstm_cell/BiasAdd_2*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
`
rnn/basic_lstm_cell/Const_8Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_4Addrnn/basic_lstm_cell/split_2:2rnn/basic_lstm_cell/Const_8*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Sigmoid_6Sigmoidrnn/basic_lstm_cell/Add_4*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_6Mulrnn/basic_lstm_cell/Add_3rnn/basic_lstm_cell/Sigmoid_6*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_7Sigmoidrnn/basic_lstm_cell/split_2*
T0*'
_output_shapes
:џџџџџџџџџ
s
rnn/basic_lstm_cell/Tanh_4Tanhrnn/basic_lstm_cell/split_2:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_7Mulrnn/basic_lstm_cell/Sigmoid_7rnn/basic_lstm_cell/Tanh_4*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_5Addrnn/basic_lstm_cell/Mul_6rnn/basic_lstm_cell/Mul_7*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/Tanh_5Tanhrnn/basic_lstm_cell/Add_5*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_8Sigmoidrnn/basic_lstm_cell/split_2:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_8Mulrnn/basic_lstm_cell/Tanh_5rnn/basic_lstm_cell/Sigmoid_8*
T0*'
_output_shapes
:џџџџџџџџџ
]
rnn/basic_lstm_cell/Const_9Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_3/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
rnn/basic_lstm_cell/concat_3ConcatV2split:3rnn/basic_lstm_cell/Mul_8!rnn/basic_lstm_cell/concat_3/axis*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_3MatMulrnn/basic_lstm_cell/concat_3rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_3BiasAddrnn/basic_lstm_cell/MatMul_3rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_10Const*
value	B :*
dtype0*
_output_shapes
: 
м
rnn/basic_lstm_cell/split_3Splitrnn/basic_lstm_cell/Const_9rnn/basic_lstm_cell/BiasAdd_3*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_11Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_6Addrnn/basic_lstm_cell/split_3:2rnn/basic_lstm_cell/Const_11*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Sigmoid_9Sigmoidrnn/basic_lstm_cell/Add_6*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_9Mulrnn/basic_lstm_cell/Add_5rnn/basic_lstm_cell/Sigmoid_9*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_10Sigmoidrnn/basic_lstm_cell/split_3*
T0*'
_output_shapes
:џџџџџџџџџ
s
rnn/basic_lstm_cell/Tanh_6Tanhrnn/basic_lstm_cell/split_3:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_10Mulrnn/basic_lstm_cell/Sigmoid_10rnn/basic_lstm_cell/Tanh_6*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_7Addrnn/basic_lstm_cell/Mul_9rnn/basic_lstm_cell/Mul_10*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/Tanh_7Tanhrnn/basic_lstm_cell/Add_7*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_11Sigmoidrnn/basic_lstm_cell/split_3:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_11Mulrnn/basic_lstm_cell/Tanh_7rnn/basic_lstm_cell/Sigmoid_11*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_12Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_4/axisConst*
dtype0*
value	B :*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_4ConcatV2split:4rnn/basic_lstm_cell/Mul_11!rnn/basic_lstm_cell/concat_4/axis*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_4MatMulrnn/basic_lstm_cell/concat_4rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_4BiasAddrnn/basic_lstm_cell/MatMul_4rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_13Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_4Splitrnn/basic_lstm_cell/Const_12rnn/basic_lstm_cell/BiasAdd_4*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_14Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_8Addrnn/basic_lstm_cell/split_4:2rnn/basic_lstm_cell/Const_14*
T0*'
_output_shapes
:џџџџџџџџџ
v
rnn/basic_lstm_cell/Sigmoid_12Sigmoidrnn/basic_lstm_cell/Add_8*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_12Mulrnn/basic_lstm_cell/Add_7rnn/basic_lstm_cell/Sigmoid_12*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_13Sigmoidrnn/basic_lstm_cell/split_4*
T0*'
_output_shapes
:џџџџџџџџџ
s
rnn/basic_lstm_cell/Tanh_8Tanhrnn/basic_lstm_cell/split_4:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_13Mulrnn/basic_lstm_cell/Sigmoid_13rnn/basic_lstm_cell/Tanh_8*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_9Addrnn/basic_lstm_cell/Mul_12rnn/basic_lstm_cell/Mul_13*
T0*'
_output_shapes
:џџџџџџџџџ
o
rnn/basic_lstm_cell/Tanh_9Tanhrnn/basic_lstm_cell/Add_9*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_14Sigmoidrnn/basic_lstm_cell/split_4:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_14Mulrnn/basic_lstm_cell/Tanh_9rnn/basic_lstm_cell/Sigmoid_14*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_15Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_5/axisConst*
value	B :*
dtype0*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_5ConcatV2split:5rnn/basic_lstm_cell/Mul_14!rnn/basic_lstm_cell/concat_5/axis*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_5MatMulrnn/basic_lstm_cell/concat_5rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_5BiasAddrnn/basic_lstm_cell/MatMul_5rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_16Const*
dtype0*
value	B :*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_5Splitrnn/basic_lstm_cell/Const_15rnn/basic_lstm_cell/BiasAdd_5*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_17Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_10Addrnn/basic_lstm_cell/split_5:2rnn/basic_lstm_cell/Const_17*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_15Sigmoidrnn/basic_lstm_cell/Add_10*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_15Mulrnn/basic_lstm_cell/Add_9rnn/basic_lstm_cell/Sigmoid_15*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_16Sigmoidrnn/basic_lstm_cell/split_5*
T0*'
_output_shapes
:џџџџџџџџџ
t
rnn/basic_lstm_cell/Tanh_10Tanhrnn/basic_lstm_cell/split_5:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_16Mulrnn/basic_lstm_cell/Sigmoid_16rnn/basic_lstm_cell/Tanh_10*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_11Addrnn/basic_lstm_cell/Mul_15rnn/basic_lstm_cell/Mul_16*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_11Tanhrnn/basic_lstm_cell/Add_11*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_17Sigmoidrnn/basic_lstm_cell/split_5:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_17Mulrnn/basic_lstm_cell/Tanh_11rnn/basic_lstm_cell/Sigmoid_17*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_18Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_6/axisConst*
value	B :*
dtype0*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_6ConcatV2split:6rnn/basic_lstm_cell/Mul_17!rnn/basic_lstm_cell/concat_6/axis*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_6MatMulrnn/basic_lstm_cell/concat_6rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_6BiasAddrnn/basic_lstm_cell/MatMul_6rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_19Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_6Splitrnn/basic_lstm_cell/Const_18rnn/basic_lstm_cell/BiasAdd_6*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_20Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 

rnn/basic_lstm_cell/Add_12Addrnn/basic_lstm_cell/split_6:2rnn/basic_lstm_cell/Const_20*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_18Sigmoidrnn/basic_lstm_cell/Add_12*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_18Mulrnn/basic_lstm_cell/Add_11rnn/basic_lstm_cell/Sigmoid_18*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_19Sigmoidrnn/basic_lstm_cell/split_6*
T0*'
_output_shapes
:џџџџџџџџџ
t
rnn/basic_lstm_cell/Tanh_12Tanhrnn/basic_lstm_cell/split_6:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_19Mulrnn/basic_lstm_cell/Sigmoid_19rnn/basic_lstm_cell/Tanh_12*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_13Addrnn/basic_lstm_cell/Mul_18rnn/basic_lstm_cell/Mul_19*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_13Tanhrnn/basic_lstm_cell/Add_13*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_20Sigmoidrnn/basic_lstm_cell/split_6:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_20Mulrnn/basic_lstm_cell/Tanh_13rnn/basic_lstm_cell/Sigmoid_20*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_21Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_7/axisConst*
value	B :*
dtype0*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_7ConcatV2split:7rnn/basic_lstm_cell/Mul_20!rnn/basic_lstm_cell/concat_7/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_7MatMulrnn/basic_lstm_cell/concat_7rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_7BiasAddrnn/basic_lstm_cell/MatMul_7rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_22Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_7Splitrnn/basic_lstm_cell/Const_21rnn/basic_lstm_cell/BiasAdd_7*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_23Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_14Addrnn/basic_lstm_cell/split_7:2rnn/basic_lstm_cell/Const_23*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_21Sigmoidrnn/basic_lstm_cell/Add_14*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_21Mulrnn/basic_lstm_cell/Add_13rnn/basic_lstm_cell/Sigmoid_21*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_22Sigmoidrnn/basic_lstm_cell/split_7*
T0*'
_output_shapes
:џџџџџџџџџ
t
rnn/basic_lstm_cell/Tanh_14Tanhrnn/basic_lstm_cell/split_7:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_22Mulrnn/basic_lstm_cell/Sigmoid_22rnn/basic_lstm_cell/Tanh_14*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_15Addrnn/basic_lstm_cell/Mul_21rnn/basic_lstm_cell/Mul_22*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_15Tanhrnn/basic_lstm_cell/Add_15*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_23Sigmoidrnn/basic_lstm_cell/split_7:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_23Mulrnn/basic_lstm_cell/Tanh_15rnn/basic_lstm_cell/Sigmoid_23*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_24Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_8/axisConst*
value	B :*
dtype0*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_8ConcatV2split:8rnn/basic_lstm_cell/Mul_23!rnn/basic_lstm_cell/concat_8/axis*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_8MatMulrnn/basic_lstm_cell/concat_8rnn/basic_lstm_cell/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_8BiasAddrnn/basic_lstm_cell/MatMul_8rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_25Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_8Splitrnn/basic_lstm_cell/Const_24rnn/basic_lstm_cell/BiasAdd_8*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_26Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_16Addrnn/basic_lstm_cell/split_8:2rnn/basic_lstm_cell/Const_26*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_24Sigmoidrnn/basic_lstm_cell/Add_16*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_24Mulrnn/basic_lstm_cell/Add_15rnn/basic_lstm_cell/Sigmoid_24*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_25Sigmoidrnn/basic_lstm_cell/split_8*
T0*'
_output_shapes
:џџџџџџџџџ
t
rnn/basic_lstm_cell/Tanh_16Tanhrnn/basic_lstm_cell/split_8:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_25Mulrnn/basic_lstm_cell/Sigmoid_25rnn/basic_lstm_cell/Tanh_16*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_17Addrnn/basic_lstm_cell/Mul_24rnn/basic_lstm_cell/Mul_25*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_17Tanhrnn/basic_lstm_cell/Add_17*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_26Sigmoidrnn/basic_lstm_cell/split_8:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_26Mulrnn/basic_lstm_cell/Tanh_17rnn/basic_lstm_cell/Sigmoid_26*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_27Const*
value	B :*
dtype0*
_output_shapes
: 
c
!rnn/basic_lstm_cell/concat_9/axisConst*
value	B :*
dtype0*
_output_shapes
: 
З
rnn/basic_lstm_cell/concat_9ConcatV2split:9rnn/basic_lstm_cell/Mul_26!rnn/basic_lstm_cell/concat_9/axis*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
Н
rnn/basic_lstm_cell/MatMul_9MatMulrnn/basic_lstm_cell/concat_9rnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџP
Ў
rnn/basic_lstm_cell/BiasAdd_9BiasAddrnn/basic_lstm_cell/MatMul_9rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_28Const*
dtype0*
value	B :*
_output_shapes
: 
н
rnn/basic_lstm_cell/split_9Splitrnn/basic_lstm_cell/Const_27rnn/basic_lstm_cell/BiasAdd_9*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_29Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_18Addrnn/basic_lstm_cell/split_9:2rnn/basic_lstm_cell/Const_29*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_27Sigmoidrnn/basic_lstm_cell/Add_18*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_27Mulrnn/basic_lstm_cell/Add_17rnn/basic_lstm_cell/Sigmoid_27*
T0*'
_output_shapes
:џџџџџџџџџ
x
rnn/basic_lstm_cell/Sigmoid_28Sigmoidrnn/basic_lstm_cell/split_9*
T0*'
_output_shapes
:џџџџџџџџџ
t
rnn/basic_lstm_cell/Tanh_18Tanhrnn/basic_lstm_cell/split_9:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_28Mulrnn/basic_lstm_cell/Sigmoid_28rnn/basic_lstm_cell/Tanh_18*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_19Addrnn/basic_lstm_cell/Mul_27rnn/basic_lstm_cell/Mul_28*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_19Tanhrnn/basic_lstm_cell/Add_19*
T0*'
_output_shapes
:џџџџџџџџџ
z
rnn/basic_lstm_cell/Sigmoid_29Sigmoidrnn/basic_lstm_cell/split_9:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_29Mulrnn/basic_lstm_cell/Tanh_19rnn/basic_lstm_cell/Sigmoid_29*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_30Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_10/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_10ConcatV2split:10rnn/basic_lstm_cell/Mul_29"rnn/basic_lstm_cell/concat_10/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_10MatMulrnn/basic_lstm_cell/concat_10rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_10BiasAddrnn/basic_lstm_cell/MatMul_10rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_31Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_10Splitrnn/basic_lstm_cell/Const_30rnn/basic_lstm_cell/BiasAdd_10*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_32Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_20Addrnn/basic_lstm_cell/split_10:2rnn/basic_lstm_cell/Const_32*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_30Sigmoidrnn/basic_lstm_cell/Add_20*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_30Mulrnn/basic_lstm_cell/Add_19rnn/basic_lstm_cell/Sigmoid_30*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_31Sigmoidrnn/basic_lstm_cell/split_10*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_20Tanhrnn/basic_lstm_cell/split_10:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_31Mulrnn/basic_lstm_cell/Sigmoid_31rnn/basic_lstm_cell/Tanh_20*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_21Addrnn/basic_lstm_cell/Mul_30rnn/basic_lstm_cell/Mul_31*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_21Tanhrnn/basic_lstm_cell/Add_21*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_32Sigmoidrnn/basic_lstm_cell/split_10:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_32Mulrnn/basic_lstm_cell/Tanh_21rnn/basic_lstm_cell/Sigmoid_32*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_33Const*
dtype0*
value	B :*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_11/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_11ConcatV2split:11rnn/basic_lstm_cell/Mul_32"rnn/basic_lstm_cell/concat_11/axis*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_11MatMulrnn/basic_lstm_cell/concat_11rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_11BiasAddrnn/basic_lstm_cell/MatMul_11rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_34Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_11Splitrnn/basic_lstm_cell/Const_33rnn/basic_lstm_cell/BiasAdd_11*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_35Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_22Addrnn/basic_lstm_cell/split_11:2rnn/basic_lstm_cell/Const_35*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_33Sigmoidrnn/basic_lstm_cell/Add_22*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_33Mulrnn/basic_lstm_cell/Add_21rnn/basic_lstm_cell/Sigmoid_33*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_34Sigmoidrnn/basic_lstm_cell/split_11*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_22Tanhrnn/basic_lstm_cell/split_11:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_34Mulrnn/basic_lstm_cell/Sigmoid_34rnn/basic_lstm_cell/Tanh_22*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_23Addrnn/basic_lstm_cell/Mul_33rnn/basic_lstm_cell/Mul_34*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_23Tanhrnn/basic_lstm_cell/Add_23*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_35Sigmoidrnn/basic_lstm_cell/split_11:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_35Mulrnn/basic_lstm_cell/Tanh_23rnn/basic_lstm_cell/Sigmoid_35*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_36Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_12/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_12ConcatV2split:12rnn/basic_lstm_cell/Mul_35"rnn/basic_lstm_cell/concat_12/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_12MatMulrnn/basic_lstm_cell/concat_12rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_12BiasAddrnn/basic_lstm_cell/MatMul_12rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_37Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_12Splitrnn/basic_lstm_cell/Const_36rnn/basic_lstm_cell/BiasAdd_12*
	num_split*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_38Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_24Addrnn/basic_lstm_cell/split_12:2rnn/basic_lstm_cell/Const_38*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_36Sigmoidrnn/basic_lstm_cell/Add_24*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_36Mulrnn/basic_lstm_cell/Add_23rnn/basic_lstm_cell/Sigmoid_36*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_37Sigmoidrnn/basic_lstm_cell/split_12*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_24Tanhrnn/basic_lstm_cell/split_12:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_37Mulrnn/basic_lstm_cell/Sigmoid_37rnn/basic_lstm_cell/Tanh_24*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_25Addrnn/basic_lstm_cell/Mul_36rnn/basic_lstm_cell/Mul_37*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_25Tanhrnn/basic_lstm_cell/Add_25*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_38Sigmoidrnn/basic_lstm_cell/split_12:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_38Mulrnn/basic_lstm_cell/Tanh_25rnn/basic_lstm_cell/Sigmoid_38*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_39Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_13/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_13ConcatV2split:13rnn/basic_lstm_cell/Mul_38"rnn/basic_lstm_cell/concat_13/axis*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_13MatMulrnn/basic_lstm_cell/concat_13rnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_13BiasAddrnn/basic_lstm_cell/MatMul_13rnn/basic_lstm_cell/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_40Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_13Splitrnn/basic_lstm_cell/Const_39rnn/basic_lstm_cell/BiasAdd_13*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_41Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_26Addrnn/basic_lstm_cell/split_13:2rnn/basic_lstm_cell/Const_41*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_39Sigmoidrnn/basic_lstm_cell/Add_26*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_39Mulrnn/basic_lstm_cell/Add_25rnn/basic_lstm_cell/Sigmoid_39*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_40Sigmoidrnn/basic_lstm_cell/split_13*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_26Tanhrnn/basic_lstm_cell/split_13:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_40Mulrnn/basic_lstm_cell/Sigmoid_40rnn/basic_lstm_cell/Tanh_26*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_27Addrnn/basic_lstm_cell/Mul_39rnn/basic_lstm_cell/Mul_40*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_27Tanhrnn/basic_lstm_cell/Add_27*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_41Sigmoidrnn/basic_lstm_cell/split_13:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_41Mulrnn/basic_lstm_cell/Tanh_27rnn/basic_lstm_cell/Sigmoid_41*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_42Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_14/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_14ConcatV2split:14rnn/basic_lstm_cell/Mul_41"rnn/basic_lstm_cell/concat_14/axis*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_14MatMulrnn/basic_lstm_cell/concat_14rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_14BiasAddrnn/basic_lstm_cell/MatMul_14rnn/basic_lstm_cell/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_43Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_14Splitrnn/basic_lstm_cell/Const_42rnn/basic_lstm_cell/BiasAdd_14*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_44Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_28Addrnn/basic_lstm_cell/split_14:2rnn/basic_lstm_cell/Const_44*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_42Sigmoidrnn/basic_lstm_cell/Add_28*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_42Mulrnn/basic_lstm_cell/Add_27rnn/basic_lstm_cell/Sigmoid_42*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_43Sigmoidrnn/basic_lstm_cell/split_14*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_28Tanhrnn/basic_lstm_cell/split_14:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_43Mulrnn/basic_lstm_cell/Sigmoid_43rnn/basic_lstm_cell/Tanh_28*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_29Addrnn/basic_lstm_cell/Mul_42rnn/basic_lstm_cell/Mul_43*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_29Tanhrnn/basic_lstm_cell/Add_29*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_44Sigmoidrnn/basic_lstm_cell/split_14:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_44Mulrnn/basic_lstm_cell/Tanh_29rnn/basic_lstm_cell/Sigmoid_44*
T0*'
_output_shapes
:џџџџџџџџџ
^
rnn/basic_lstm_cell/Const_45Const*
value	B :*
dtype0*
_output_shapes
: 
d
"rnn/basic_lstm_cell/concat_15/axisConst*
value	B :*
dtype0*
_output_shapes
: 
К
rnn/basic_lstm_cell/concat_15ConcatV2split:15rnn/basic_lstm_cell/Mul_44"rnn/basic_lstm_cell/concat_15/axis*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџ
П
rnn/basic_lstm_cell/MatMul_15MatMulrnn/basic_lstm_cell/concat_15rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџP
А
rnn/basic_lstm_cell/BiasAdd_15BiasAddrnn/basic_lstm_cell/MatMul_15rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџP
^
rnn/basic_lstm_cell/Const_46Const*
value	B :*
dtype0*
_output_shapes
: 
п
rnn/basic_lstm_cell/split_15Splitrnn/basic_lstm_cell/Const_45rnn/basic_lstm_cell/BiasAdd_15*
T0*
	num_split*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
a
rnn/basic_lstm_cell/Const_47Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

rnn/basic_lstm_cell/Add_30Addrnn/basic_lstm_cell/split_15:2rnn/basic_lstm_cell/Const_47*
T0*'
_output_shapes
:џџџџџџџџџ
w
rnn/basic_lstm_cell/Sigmoid_45Sigmoidrnn/basic_lstm_cell/Add_30*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_45Mulrnn/basic_lstm_cell/Add_29rnn/basic_lstm_cell/Sigmoid_45*
T0*'
_output_shapes
:џџџџџџџџџ
y
rnn/basic_lstm_cell/Sigmoid_46Sigmoidrnn/basic_lstm_cell/split_15*
T0*'
_output_shapes
:џџџџџџџџџ
u
rnn/basic_lstm_cell/Tanh_30Tanhrnn/basic_lstm_cell/split_15:1*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_46Mulrnn/basic_lstm_cell/Sigmoid_46rnn/basic_lstm_cell/Tanh_30*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Add_31Addrnn/basic_lstm_cell/Mul_45rnn/basic_lstm_cell/Mul_46*
T0*'
_output_shapes
:џџџџџџџџџ
q
rnn/basic_lstm_cell/Tanh_31Tanhrnn/basic_lstm_cell/Add_31*
T0*'
_output_shapes
:џџџџџџџџџ
{
rnn/basic_lstm_cell/Sigmoid_47Sigmoidrnn/basic_lstm_cell/split_15:3*
T0*'
_output_shapes
:џџџџџџџџџ

rnn/basic_lstm_cell/Mul_47Mulrnn/basic_lstm_cell/Tanh_31rnn/basic_lstm_cell/Sigmoid_47*
T0*'
_output_shapes
:џџџџџџџџџ

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *   П*
_class
loc:@dense/kernel*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *   ?*
_class
loc:@dense/kernel*
_output_shapes
: 
щ
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
dtype0*
seed2м*
seedвЎК	*
_output_shapes

:
Ю
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
: 
р
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
в
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes

:
Ё
dense/kernel
VariableV2*
	container *
shape
:*
shared_name *
_class
loc:@dense/kernel*
dtype0*
_output_shapes

:
Ч
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
u
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes

:

dense/bias/Initializer/zerosConst*
dtype0*
valueB*    *
_class
loc:@dense/bias*
_output_shapes
:


dense/bias
VariableV2*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
shape:*
_output_shapes
:
В
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
k
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes
:

dense/MatMulMatMulrnn/basic_lstm_cell/Mul_47dense/kernel/read*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:џџџџџџџџџ

dense/BiasAddBiasAdddense/MatMuldense/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ

$mean_squared_error/SquaredDifferenceSquaredDifferencedense/BiasAddIteratorGetNext:1*
T0*'
_output_shapes
:џџџџџџџџџ
t
/mean_squared_error/assert_broadcastable/weightsConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
x
5mean_squared_error/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
v
4mean_squared_error/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 

4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
u
3mean_squared_error/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
K
Cmean_squared_error/assert_broadcastable/static_scalar_check_successNoOp
Љ
mean_squared_error/ToFloat_3/xConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *  ?*
_output_shapes
: 

mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/ToFloat_3/x*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
mean_squared_error/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:

mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Б
&mean_squared_error/num_present/Equal/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

$mean_squared_error/num_present/EqualEqualmean_squared_error/ToFloat_3/x&mean_squared_error/num_present/Equal/y*
T0*
_output_shapes
: 
Д
)mean_squared_error/num_present/zeros_likeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
З
.mean_squared_error/num_present/ones_like/ShapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Й
.mean_squared_error/num_present/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 
У
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*

index_type0*
T0*
_output_shapes
: 
Ы
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
м
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 
к
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
ќ
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
й
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
Џ
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpD^mean_squared_error/assert_broadcastable/static_scalar_check_success
Ю
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
Џ
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
Ь
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:џџџџџџџџџ
Л
$mean_squared_error/num_present/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
Л
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ѓ
mean_squared_error/Const_1ConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB *
_output_shapes
: 

mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ї
mean_squared_error/Greater/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
valueB
 *    *
_output_shapes
: 

mean_squared_error/GreaterGreatermean_squared_error/num_presentmean_squared_error/Greater/y*
T0*
_output_shapes
: 
Ѕ
mean_squared_error/Equal/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
~
mean_squared_error/EqualEqualmean_squared_error/num_presentmean_squared_error/Equal/y*
T0*
_output_shapes
: 
Ћ
"mean_squared_error/ones_like/ShapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
­
"mean_squared_error/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  ?*
dtype0*
_output_shapes
: 

mean_squared_error/ones_likeFill"mean_squared_error/ones_like/Shape"mean_squared_error/ones_like/Const*
T0*

index_type0*
_output_shapes
: 

mean_squared_error/SelectSelectmean_squared_error/Equalmean_squared_error/ones_likemean_squared_error/num_present*
T0*
_output_shapes
: 
w
mean_squared_error/divRealDivmean_squared_error/Sum_1mean_squared_error/Select*
T0*
_output_shapes
: 
Ј
mean_squared_error/zeros_likeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 

mean_squared_error/valueSelectmean_squared_error/Greatermean_squared_error/divmean_squared_error/zeros_like*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
w
2gradients/mean_squared_error/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Й
.gradients/mean_squared_error/value_grad/SelectSelectmean_squared_error/Greatergradients/Fill2gradients/mean_squared_error/value_grad/zeros_like*
T0*
_output_shapes
: 
Л
0gradients/mean_squared_error/value_grad/Select_1Selectmean_squared_error/Greater2gradients/mean_squared_error/value_grad/zeros_likegradients/Fill*
T0*
_output_shapes
: 
Є
8gradients/mean_squared_error/value_grad/tuple/group_depsNoOp/^gradients/mean_squared_error/value_grad/Select1^gradients/mean_squared_error/value_grad/Select_1

@gradients/mean_squared_error/value_grad/tuple/control_dependencyIdentity.gradients/mean_squared_error/value_grad/Select9^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/mean_squared_error/value_grad/Select*
_output_shapes
: 
Ё
Bgradients/mean_squared_error/value_grad/tuple/control_dependency_1Identity0gradients/mean_squared_error/value_grad/Select_19^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/mean_squared_error/value_grad/Select_1*
_output_shapes
: 
n
+gradients/mean_squared_error/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
p
-gradients/mean_squared_error/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
э
;gradients/mean_squared_error/div_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/div_grad/Shape-gradients/mean_squared_error/div_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ж
-gradients/mean_squared_error/div_grad/RealDivRealDiv@gradients/mean_squared_error/value_grad/tuple/control_dependencymean_squared_error/Select*
T0*
_output_shapes
: 
м
)gradients/mean_squared_error/div_grad/SumSum-gradients/mean_squared_error/div_grad/RealDiv;gradients/mean_squared_error/div_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
П
-gradients/mean_squared_error/div_grad/ReshapeReshape)gradients/mean_squared_error/div_grad/Sum+gradients/mean_squared_error/div_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
k
)gradients/mean_squared_error/div_grad/NegNegmean_squared_error/Sum_1*
T0*
_output_shapes
: 
Ё
/gradients/mean_squared_error/div_grad/RealDiv_1RealDiv)gradients/mean_squared_error/div_grad/Negmean_squared_error/Select*
T0*
_output_shapes
: 
Ї
/gradients/mean_squared_error/div_grad/RealDiv_2RealDiv/gradients/mean_squared_error/div_grad/RealDiv_1mean_squared_error/Select*
T0*
_output_shapes
: 
Ф
)gradients/mean_squared_error/div_grad/mulMul@gradients/mean_squared_error/value_grad/tuple/control_dependency/gradients/mean_squared_error/div_grad/RealDiv_2*
T0*
_output_shapes
: 
м
+gradients/mean_squared_error/div_grad/Sum_1Sum)gradients/mean_squared_error/div_grad/mul=gradients/mean_squared_error/div_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Х
/gradients/mean_squared_error/div_grad/Reshape_1Reshape+gradients/mean_squared_error/div_grad/Sum_1-gradients/mean_squared_error/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
 
6gradients/mean_squared_error/div_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/div_grad/Reshape0^gradients/mean_squared_error/div_grad/Reshape_1

>gradients/mean_squared_error/div_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/div_grad/Reshape7^gradients/mean_squared_error/div_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/mean_squared_error/div_grad/Reshape*
_output_shapes
: 

@gradients/mean_squared_error/div_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/div_grad/Reshape_17^gradients/mean_squared_error/div_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/div_grad/Reshape_1*
_output_shapes
: 
x
5gradients/mean_squared_error/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
р
/gradients/mean_squared_error/Sum_1_grad/ReshapeReshape>gradients/mean_squared_error/div_grad/tuple/control_dependency5gradients/mean_squared_error/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
p
-gradients/mean_squared_error/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
Ч
,gradients/mean_squared_error/Sum_1_grad/TileTile/gradients/mean_squared_error/Sum_1_grad/Reshape-gradients/mean_squared_error/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
x
3gradients/mean_squared_error/Select_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ы
/gradients/mean_squared_error/Select_grad/SelectSelectmean_squared_error/Equal@gradients/mean_squared_error/div_grad/tuple/control_dependency_13gradients/mean_squared_error/Select_grad/zeros_like*
T0*
_output_shapes
: 
э
1gradients/mean_squared_error/Select_grad/Select_1Selectmean_squared_error/Equal3gradients/mean_squared_error/Select_grad/zeros_like@gradients/mean_squared_error/div_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
Ї
9gradients/mean_squared_error/Select_grad/tuple/group_depsNoOp0^gradients/mean_squared_error/Select_grad/Select2^gradients/mean_squared_error/Select_grad/Select_1

Agradients/mean_squared_error/Select_grad/tuple/control_dependencyIdentity/gradients/mean_squared_error/Select_grad/Select:^gradients/mean_squared_error/Select_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/Select_grad/Select*
_output_shapes
: 
Ѕ
Cgradients/mean_squared_error/Select_grad/tuple/control_dependency_1Identity1gradients/mean_squared_error/Select_grad/Select_1:^gradients/mean_squared_error/Select_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/mean_squared_error/Select_grad/Select_1*
_output_shapes
: 

3gradients/mean_squared_error/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
в
-gradients/mean_squared_error/Sum_grad/ReshapeReshape,gradients/mean_squared_error/Sum_1_grad/Tile3gradients/mean_squared_error/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:

+gradients/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
T0*
out_type0*
_output_shapes
:
в
*gradients/mean_squared_error/Sum_grad/TileTile-gradients/mean_squared_error/Sum_grad/Reshape+gradients/mean_squared_error/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ

+gradients/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
p
-gradients/mean_squared_error/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
э
;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/Mul_grad/Shape-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ў
)gradients/mean_squared_error/Mul_grad/MulMul*gradients/mean_squared_error/Sum_grad/Tilemean_squared_error/ToFloat_3/x*
T0*'
_output_shapes
:џџџџџџџџџ
и
)gradients/mean_squared_error/Mul_grad/SumSum)gradients/mean_squared_error/Mul_grad/Mul;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
а
-gradients/mean_squared_error/Mul_grad/ReshapeReshape)gradients/mean_squared_error/Mul_grad/Sum+gradients/mean_squared_error/Mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ж
+gradients/mean_squared_error/Mul_grad/Mul_1Mul$mean_squared_error/SquaredDifference*gradients/mean_squared_error/Sum_grad/Tile*
T0*'
_output_shapes
:џџџџџџџџџ
о
+gradients/mean_squared_error/Mul_grad/Sum_1Sum+gradients/mean_squared_error/Mul_grad/Mul_1=gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Х
/gradients/mean_squared_error/Mul_grad/Reshape_1Reshape+gradients/mean_squared_error/Mul_grad/Sum_1-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
 
6gradients/mean_squared_error/Mul_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/Mul_grad/Reshape0^gradients/mean_squared_error/Mul_grad/Reshape_1
І
>gradients/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/Mul_grad/Reshape7^gradients/mean_squared_error/Mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/mean_squared_error/Mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

@gradients/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/Mul_grad/Reshape_17^gradients/mean_squared_error/Mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/Mul_grad/Reshape_1*
_output_shapes
: 

;gradients/mean_squared_error/num_present_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
љ
5gradients/mean_squared_error/num_present_grad/ReshapeReshapeCgradients/mean_squared_error/Select_grad/tuple/control_dependency_1;gradients/mean_squared_error/num_present_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes

:
Ѓ
3gradients/mean_squared_error/num_present_grad/ShapeShape0mean_squared_error/num_present/broadcast_weights*
T0*
out_type0*
_output_shapes
:
ъ
2gradients/mean_squared_error/num_present_grad/TileTile5gradients/mean_squared_error/num_present_grad/Reshape3gradients/mean_squared_error/num_present_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ

Egradients/mean_squared_error/num_present/broadcast_weights_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
С
Ggradients/mean_squared_error/num_present/broadcast_weights_grad/Shape_1Shape:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*
out_type0*
_output_shapes
:
Л
Ugradients/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgsBroadcastGradientArgsEgradients/mean_squared_error/num_present/broadcast_weights_grad/ShapeGgradients/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ь
Cgradients/mean_squared_error/num_present/broadcast_weights_grad/MulMul2gradients/mean_squared_error/num_present_grad/Tile:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:џџџџџџџџџ
І
Cgradients/mean_squared_error/num_present/broadcast_weights_grad/SumSumCgradients/mean_squared_error/num_present/broadcast_weights_grad/MulUgradients/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:

Ggradients/mean_squared_error/num_present/broadcast_weights_grad/ReshapeReshapeCgradients/mean_squared_error/num_present/broadcast_weights_grad/SumEgradients/mean_squared_error/num_present/broadcast_weights_grad/Shape*
Tshape0*
T0*
_output_shapes
: 
й
Egradients/mean_squared_error/num_present/broadcast_weights_grad/Mul_1Mul%mean_squared_error/num_present/Select2gradients/mean_squared_error/num_present_grad/Tile*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
Egradients/mean_squared_error/num_present/broadcast_weights_grad/Sum_1SumEgradients/mean_squared_error/num_present/broadcast_weights_grad/Mul_1Wgradients/mean_squared_error/num_present/broadcast_weights_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Є
Igradients/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1ReshapeEgradients/mean_squared_error/num_present/broadcast_weights_grad/Sum_1Ggradients/mean_squared_error/num_present/broadcast_weights_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ю
Pgradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_depsNoOpH^gradients/mean_squared_error/num_present/broadcast_weights_grad/ReshapeJ^gradients/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1
§
Xgradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependencyIdentityGgradients/mean_squared_error/num_present/broadcast_weights_grad/ReshapeQ^gradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/mean_squared_error/num_present/broadcast_weights_grad/Reshape*
_output_shapes
: 

Zgradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1IdentityIgradients/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1Q^gradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/mean_squared_error/num_present/broadcast_weights_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
 
Ogradients/mean_squared_error/num_present/broadcast_weights/ones_like_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
П
Mgradients/mean_squared_error/num_present/broadcast_weights/ones_like_grad/SumSumZgradients/mean_squared_error/num_present/broadcast_weights_grad/tuple/control_dependency_1Ogradients/mean_squared_error/num_present/broadcast_weights/ones_like_grad/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 

9gradients/mean_squared_error/SquaredDifference_grad/ShapeShapedense/BiasAdd*
out_type0*
T0*
_output_shapes
:

;gradients/mean_squared_error/SquaredDifference_grad/Shape_1ShapeIteratorGetNext:1*
T0*
out_type0*
_output_shapes
:

Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/mean_squared_error/SquaredDifference_grad/Shape;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Р
:gradients/mean_squared_error/SquaredDifference_grad/scalarConst?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
dtype0*
valueB
 *   @*
_output_shapes
: 
ь
7gradients/mean_squared_error/SquaredDifference_grad/mulMul:gradients/mean_squared_error/SquaredDifference_grad/scalar>gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
г
7gradients/mean_squared_error/SquaredDifference_grad/subSubdense/BiasAddIteratorGetNext:1?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ф
9gradients/mean_squared_error/SquaredDifference_grad/mul_1Mul7gradients/mean_squared_error/SquaredDifference_grad/mul7gradients/mean_squared_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:џџџџџџџџџ

7gradients/mean_squared_error/SquaredDifference_grad/SumSum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
њ
;gradients/mean_squared_error/SquaredDifference_grad/ReshapeReshape7gradients/mean_squared_error/SquaredDifference_grad/Sum9gradients/mean_squared_error/SquaredDifference_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ

9gradients/mean_squared_error/SquaredDifference_grad/Sum_1Sum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Kgradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:

=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape9gradients/mean_squared_error/SquaredDifference_grad/Sum_1;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Џ
7gradients/mean_squared_error/SquaredDifference_grad/NegNeg=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
Dgradients/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp8^gradients/mean_squared_error/SquaredDifference_grad/Neg<^gradients/mean_squared_error/SquaredDifference_grad/Reshape
о
Lgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity;gradients/mean_squared_error/SquaredDifference_grad/ReshapeE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
и
Ngradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity7gradients/mean_squared_error/SquaredDifference_grad/NegE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/mean_squared_error/SquaredDifference_grad/Neg*'
_output_shapes
:џџџџџџџџџ
С
(gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGradLgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:
Џ
-gradients/dense/BiasAdd_grad/tuple/group_depsNoOp)^gradients/dense/BiasAdd_grad/BiasAddGradM^gradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency
С
5gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentityLgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
џ
7gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/dense/BiasAdd_grad/BiasAddGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ю
"gradients/dense/MatMul_grad/MatMulMatMul5gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
а
$gradients/dense/MatMul_grad/MatMul_1MatMulrnn/basic_lstm_cell/Mul_475gradients/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:

,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
ќ
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
љ
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes

:

/gradients/rnn/basic_lstm_cell/Mul_47_grad/ShapeShapernn/basic_lstm_cell/Tanh_31*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_47_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_47*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_47_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_47_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_47_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
М
-gradients/rnn/basic_lstm_cell/Mul_47_grad/MulMul4gradients/dense/MatMul_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_47*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_47_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_47_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_47_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_47_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_47_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_47_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Л
/gradients/rnn/basic_lstm_cell/Mul_47_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_314gradients/dense/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_47_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_47_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_47_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_47_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_47_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_47_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_31_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_31Bgradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_47_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_47Dgradients/rnn/basic_lstm_cell/Mul_47_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_31_grad/ShapeShapernn/basic_lstm_cell/Mul_45*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_31_grad/Shape_1Shapernn/basic_lstm_cell/Mul_46*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_31_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_31_grad/Shape1gradients/rnn/basic_lstm_cell/Add_31_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ъ
-gradients/rnn/basic_lstm_cell/Add_31_grad/SumSum3gradients/rnn/basic_lstm_cell/Tanh_31_grad/TanhGrad?gradients/rnn/basic_lstm_cell/Add_31_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_31_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_31_grad/Sum/gradients/rnn/basic_lstm_cell/Add_31_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ю
/gradients/rnn/basic_lstm_cell/Add_31_grad/Sum_1Sum3gradients/rnn/basic_lstm_cell/Tanh_31_grad/TanhGradAgradients/rnn/basic_lstm_cell/Add_31_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_31_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_31_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_31_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_31_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_31_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_31_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_45_grad/ShapeShapernn/basic_lstm_cell/Add_29*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_45_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_45*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_45_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_45_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_45_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_45_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_45*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_45_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_45_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_45_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_45_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_45_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_45_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_45_grad/Mul_1Mulrnn/basic_lstm_cell/Add_29Bgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_45_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_45_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_45_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_45_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_45_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_46_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_46*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_46_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_30*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_46_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_46_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_46_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_46_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_30*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_46_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_46_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_46_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_46_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_46_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_46_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_46_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_46Dgradients/rnn/basic_lstm_cell/Add_31_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_46_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_46_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_46_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_46_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_46_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_46_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_45_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_45Dgradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_46_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_46Bgradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_30_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_30Dgradients/rnn/basic_lstm_cell/Mul_46_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_30_grad/ShapeShapernn/basic_lstm_cell/split_15:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_30_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_30_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_30_grad/Shape1gradients/rnn/basic_lstm_cell/Add_30_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_30_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_45_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_30_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_30_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_30_grad/Sum/gradients/rnn/basic_lstm_cell/Add_30_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_30_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_45_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_30_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_30_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_30_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_30_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_30_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_30_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_30_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_30_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_30_grad/Reshape_1*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_15_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_46_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_30_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_30_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_47_grad/SigmoidGradrnn/basic_lstm_cell/Const_45*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_15_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_15_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_15_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_15_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/group_deps*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/BiasAddGrad*
T0*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_15Fgradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_15_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_15_grad/modFloorMod"rnn/basic_lstm_cell/concat_15/axis1gradients/rnn/basic_lstm_cell/concat_15_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeShapesplit:15*
T0*
out_type0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeNShapeNsplit:15rnn/basic_lstm_cell/Mul_44*
out_type0*
N*
T0* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_15_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_15_grad/mod3gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_15_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_15_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_15_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_15_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_15_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_15_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_15_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_15_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_15_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_15_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_15_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_15_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_15_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_15_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_15_grad/tuple/group_deps*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_15_grad/Slice_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_44_grad/ShapeShapernn/basic_lstm_cell/Tanh_29*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_44_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_44*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_44_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_44_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_44_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_44_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_15_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_44*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_44_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_44_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_44_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_44_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_44_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_44_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_44_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_29Ggradients/rnn/basic_lstm_cell/concat_15_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_44_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_44_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_44_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_44_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_44_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_44_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_29_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_29Bgradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_44_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_44Dgradients/rnn/basic_lstm_cell/Mul_44_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddNAddNBgradients/rnn/basic_lstm_cell/Mul_45_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_29_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_45_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_29_grad/ShapeShapernn/basic_lstm_cell/Mul_42*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_29_grad/Shape_1Shapernn/basic_lstm_cell/Mul_43*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_29_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_29_grad/Shape1gradients/rnn/basic_lstm_cell/Add_29_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Х
-gradients/rnn/basic_lstm_cell/Add_29_grad/SumSumgradients/AddN?gradients/rnn/basic_lstm_cell/Add_29_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_29_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_29_grad/Sum/gradients/rnn/basic_lstm_cell/Add_29_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Щ
/gradients/rnn/basic_lstm_cell/Add_29_grad/Sum_1Sumgradients/AddNAgradients/rnn/basic_lstm_cell/Add_29_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_29_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_29_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_29_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_29_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_29_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_29_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_42_grad/ShapeShapernn/basic_lstm_cell/Add_27*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_42_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_42*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_42_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_42_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_42_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_42_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_42*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_42_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_42_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_42_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_42_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_42_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_42_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_42_grad/Mul_1Mulrnn/basic_lstm_cell/Add_27Bgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_42_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_42_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_42_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_42_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_42_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_43_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_43*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_43_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_28*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_43_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_43_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_43_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_43_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_28*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_43_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_43_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_43_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_43_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_43_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_43_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_43_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_43Dgradients/rnn/basic_lstm_cell/Add_29_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_43_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_43_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_43_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_43_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_43_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_43_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_42_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_42Dgradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_43_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_43Bgradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_28_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_28Dgradients/rnn/basic_lstm_cell/Mul_43_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_28_grad/ShapeShapernn/basic_lstm_cell/split_14:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_28_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_28_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_28_grad/Shape1gradients/rnn/basic_lstm_cell/Add_28_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_28_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_42_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_28_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_28_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_28_grad/Sum/gradients/rnn/basic_lstm_cell/Add_28_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_28_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_42_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_28_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_28_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_28_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_28_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_28_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_28_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_28_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_28_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_28_grad/Reshape_1*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_14_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_43_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_28_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_28_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_44_grad/SigmoidGradrnn/basic_lstm_cell/Const_42*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_14_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_14_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_14_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_14_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_14_grad/BiasAddGrad*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_14Fgradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_14_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_14_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_14_grad/modFloorMod"rnn/basic_lstm_cell/concat_14/axis1gradients/rnn/basic_lstm_cell/concat_14_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeShapesplit:14*
T0*
out_type0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeNShapeNsplit:14rnn/basic_lstm_cell/Mul_41*
T0*
out_type0*
N* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_14_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_14_grad/mod3gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_14_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_14_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_14_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_14_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_14_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_14_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_14_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_14_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_14_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_14_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_14_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_14_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_14_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_14_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_14_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_14_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_41_grad/ShapeShapernn/basic_lstm_cell/Tanh_27*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_41_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_41*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_41_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_41_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_41_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_41_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_14_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_41*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_41_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_41_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_41_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_41_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_41_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_41_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_41_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_27Ggradients/rnn/basic_lstm_cell/concat_14_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_41_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_41_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_41_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_41_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_41_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_41_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_27_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_27Bgradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_41_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_41Dgradients/rnn/basic_lstm_cell/Mul_41_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_1AddNBgradients/rnn/basic_lstm_cell/Mul_42_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_27_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_42_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_27_grad/ShapeShapernn/basic_lstm_cell/Mul_39*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_27_grad/Shape_1Shapernn/basic_lstm_cell/Mul_40*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_27_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_27_grad/Shape1gradients/rnn/basic_lstm_cell/Add_27_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_27_grad/SumSumgradients/AddN_1?gradients/rnn/basic_lstm_cell/Add_27_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_27_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_27_grad/Sum/gradients/rnn/basic_lstm_cell/Add_27_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_27_grad/Sum_1Sumgradients/AddN_1Agradients/rnn/basic_lstm_cell/Add_27_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_27_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_27_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_27_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_27_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_27_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_27_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_39_grad/ShapeShapernn/basic_lstm_cell/Add_25*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_39_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_39*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_39_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_39_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_39_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_39_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_39*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_39_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_39_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_39_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_39_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_39_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_39_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_39_grad/Mul_1Mulrnn/basic_lstm_cell/Add_25Bgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_39_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_39_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_39_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_39_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_39_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_40_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_40*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_40_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_26*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_40_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_40_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_40_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_40_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_26*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_40_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_40_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_40_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_40_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_40_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_40_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_40_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_40Dgradients/rnn/basic_lstm_cell/Add_27_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_40_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_40_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_40_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_40_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_40_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_40_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_39_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_39Dgradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_40_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_40Bgradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_26_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_26Dgradients/rnn/basic_lstm_cell/Mul_40_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_26_grad/ShapeShapernn/basic_lstm_cell/split_13:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_26_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_26_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_26_grad/Shape1gradients/rnn/basic_lstm_cell/Add_26_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_26_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_39_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_26_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_26_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_26_grad/Sum/gradients/rnn/basic_lstm_cell/Add_26_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_26_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_39_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_26_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_26_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_26_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_26_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_26_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_26_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_26_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_26_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_26_grad/Reshape_1*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_13_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_40_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_26_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_26_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_41_grad/SigmoidGradrnn/basic_lstm_cell/Const_39*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_13_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_13_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_13_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_13_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_13_grad/BiasAddGrad*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_13Fgradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_13_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_13_grad/RankConst*
dtype0*
value	B :*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_13_grad/modFloorMod"rnn/basic_lstm_cell/concat_13/axis1gradients/rnn/basic_lstm_cell/concat_13_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeShapesplit:13*
T0*
out_type0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeNShapeNsplit:13rnn/basic_lstm_cell/Mul_38*
T0*
out_type0*
N* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_13_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_13_grad/mod3gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_13_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_13_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_13_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_13_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_13_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_13_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_13_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_13_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_13_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_13_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_13_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_13_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_13_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_13_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_13_grad/tuple/group_deps*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_13_grad/Slice_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_38_grad/ShapeShapernn/basic_lstm_cell/Tanh_25*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_38_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_38*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_38_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_38_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_38_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_38_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_13_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_38*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_38_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_38_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_38_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_38_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_38_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_38_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_38_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_25Ggradients/rnn/basic_lstm_cell/concat_13_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_38_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_38_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_38_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_38_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_38_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_38_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_25_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_25Bgradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_38_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_38Dgradients/rnn/basic_lstm_cell/Mul_38_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_2AddNBgradients/rnn/basic_lstm_cell/Mul_39_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_25_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_39_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_25_grad/ShapeShapernn/basic_lstm_cell/Mul_36*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_25_grad/Shape_1Shapernn/basic_lstm_cell/Mul_37*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_25_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_25_grad/Shape1gradients/rnn/basic_lstm_cell/Add_25_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_25_grad/SumSumgradients/AddN_2?gradients/rnn/basic_lstm_cell/Add_25_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_25_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_25_grad/Sum/gradients/rnn/basic_lstm_cell/Add_25_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_25_grad/Sum_1Sumgradients/AddN_2Agradients/rnn/basic_lstm_cell/Add_25_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_25_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_25_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_25_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_25_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_25_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_25_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_36_grad/ShapeShapernn/basic_lstm_cell/Add_23*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_36_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_36*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_36_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_36_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_36_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_36_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_36*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_36_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_36_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_36_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_36_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_36_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_36_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_36_grad/Mul_1Mulrnn/basic_lstm_cell/Add_23Bgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_36_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_36_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_36_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_36_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_36_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_37_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_37*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_37_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_24*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_37_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_37_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_37_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_37_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_24*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_37_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_37_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_37_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_37_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_37_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_37_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_37_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_37Dgradients/rnn/basic_lstm_cell/Add_25_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_37_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_37_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_37_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_37_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_37_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_37_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_36_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_36Dgradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_37_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_37Bgradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_24_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_24Dgradients/rnn/basic_lstm_cell/Mul_37_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_24_grad/ShapeShapernn/basic_lstm_cell/split_12:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_24_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_24_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_24_grad/Shape1gradients/rnn/basic_lstm_cell/Add_24_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_24_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_36_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_24_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_24_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_24_grad/Sum/gradients/rnn/basic_lstm_cell/Add_24_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_24_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_36_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_24_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_24_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_24_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_24_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_24_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_24_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_24_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_24_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_24_grad/Reshape_1*
T0*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_12_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_37_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_24_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_24_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_38_grad/SigmoidGradrnn/basic_lstm_cell/Const_36*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_12_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_12_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_12_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/group_deps*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_12_grad/concat*
T0*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_12_grad/BiasAddGrad*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_12Fgradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_12_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_12_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_12_grad/modFloorMod"rnn/basic_lstm_cell/concat_12/axis1gradients/rnn/basic_lstm_cell/concat_12_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeShapesplit:12*
out_type0*
T0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeNShapeNsplit:12rnn/basic_lstm_cell/Mul_35*
T0*
out_type0*
N* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_12_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_12_grad/mod3gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_12_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_12_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_12_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_12_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_12_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_12_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_12_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_12_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_12_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_12_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_12_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_12_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_12_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_12_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_12_grad/tuple/group_deps*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_12_grad/Slice_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_35_grad/ShapeShapernn/basic_lstm_cell/Tanh_23*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_35_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_35*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_35_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_35_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_35_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_35_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_12_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_35*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_35_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_35_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_35_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_35_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_35_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_35_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_35_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_23Ggradients/rnn/basic_lstm_cell/concat_12_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_35_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_35_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_35_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_35_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_35_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_35_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_23_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_23Bgradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_35_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_35Dgradients/rnn/basic_lstm_cell/Mul_35_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_3AddNBgradients/rnn/basic_lstm_cell/Mul_36_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_23_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_36_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_23_grad/ShapeShapernn/basic_lstm_cell/Mul_33*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_23_grad/Shape_1Shapernn/basic_lstm_cell/Mul_34*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_23_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_23_grad/Shape1gradients/rnn/basic_lstm_cell/Add_23_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_23_grad/SumSumgradients/AddN_3?gradients/rnn/basic_lstm_cell/Add_23_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_23_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_23_grad/Sum/gradients/rnn/basic_lstm_cell/Add_23_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_23_grad/Sum_1Sumgradients/AddN_3Agradients/rnn/basic_lstm_cell/Add_23_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_23_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_23_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_23_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_23_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_23_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_23_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_33_grad/ShapeShapernn/basic_lstm_cell/Add_21*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_33_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_33*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_33_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_33_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_33_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_33_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_33*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_33_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_33_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_33_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_33_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_33_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_33_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_33_grad/Mul_1Mulrnn/basic_lstm_cell/Add_21Bgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_33_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_33_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_33_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_33_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_33_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_34_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_34*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_34_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_22*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_34_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_34_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_34_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_34_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_22*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_34_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_34_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_34_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_34_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_34_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_34_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_34_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_34Dgradients/rnn/basic_lstm_cell/Add_23_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_34_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_34_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_34_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_34_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_34_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_34_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_33_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_33Dgradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_34_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_34Bgradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_22_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_22Dgradients/rnn/basic_lstm_cell/Mul_34_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_22_grad/ShapeShapernn/basic_lstm_cell/split_11:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_22_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_22_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_22_grad/Shape1gradients/rnn/basic_lstm_cell/Add_22_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_22_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_33_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_22_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_22_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_22_grad/Sum/gradients/rnn/basic_lstm_cell/Add_22_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_22_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_33_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_22_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_22_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_22_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_22_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_22_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_22_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_22_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_22_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_22_grad/Reshape_1*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_11_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_34_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_22_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_22_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_35_grad/SigmoidGradrnn/basic_lstm_cell/Const_33*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_11_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_11_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_11_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_11_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_11_grad/BiasAddGrad*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_11Fgradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_11_grad/MatMul_1*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_11_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_11_grad/modFloorMod"rnn/basic_lstm_cell/concat_11/axis1gradients/rnn/basic_lstm_cell/concat_11_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeShapesplit:11*
T0*
out_type0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeNShapeNsplit:11rnn/basic_lstm_cell/Mul_32*
T0*
out_type0*
N* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_11_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_11_grad/mod3gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_11_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_11_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_11_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_11_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_11_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_11_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_11_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_11_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_11_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_11_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_11_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_11_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_11_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_11_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_11_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_11_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_32_grad/ShapeShapernn/basic_lstm_cell/Tanh_21*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_32_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_32*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_32_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_32_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_32_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_32_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_11_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_32*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_32_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_32_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_32_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_32_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_32_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_32_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_32_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_21Ggradients/rnn/basic_lstm_cell/concat_11_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_32_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_32_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_32_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_32_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_32_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_32_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_21_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_21Bgradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_32_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_32Dgradients/rnn/basic_lstm_cell/Mul_32_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_4AddNBgradients/rnn/basic_lstm_cell/Mul_33_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_21_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_33_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_21_grad/ShapeShapernn/basic_lstm_cell/Mul_30*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_21_grad/Shape_1Shapernn/basic_lstm_cell/Mul_31*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_21_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_21_grad/Shape1gradients/rnn/basic_lstm_cell/Add_21_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_21_grad/SumSumgradients/AddN_4?gradients/rnn/basic_lstm_cell/Add_21_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_21_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_21_grad/Sum/gradients/rnn/basic_lstm_cell/Add_21_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_21_grad/Sum_1Sumgradients/AddN_4Agradients/rnn/basic_lstm_cell/Add_21_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_21_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_21_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_21_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_21_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_21_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_21_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_30_grad/ShapeShapernn/basic_lstm_cell/Add_19*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_30_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_30*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_30_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_30_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_30_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_30_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_30*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_30_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_30_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_30_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_30_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_30_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_30_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_30_grad/Mul_1Mulrnn/basic_lstm_cell/Add_19Bgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_30_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_30_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_30_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_30_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_30_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_31_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_31*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_31_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_20*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_31_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_31_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_31_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_31_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_20*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_31_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_31_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_31_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_31_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_31_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_31_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_31_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_31Dgradients/rnn/basic_lstm_cell/Add_21_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_31_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_31_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_31_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_31_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_31_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_31_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_30_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_30Dgradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_31_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_31Bgradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_20_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_20Dgradients/rnn/basic_lstm_cell/Mul_31_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_20_grad/ShapeShapernn/basic_lstm_cell/split_10:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_20_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_20_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_20_grad/Shape1gradients/rnn/basic_lstm_cell/Add_20_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_20_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_30_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_20_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_20_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_20_grad/Sum/gradients/rnn/basic_lstm_cell/Add_20_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_20_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_30_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_20_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_20_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_20_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_20_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_20_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_20_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_20_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_20_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_20_grad/Reshape_1*
_output_shapes
: 

2gradients/rnn/basic_lstm_cell/split_10_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_31_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_20_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_20_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_32_grad/SigmoidGradrnn/basic_lstm_cell/Const_30*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџP
И
9gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/BiasAddGradBiasAddGrad2gradients/rnn/basic_lstm_cell/split_10_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
З
>gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/group_depsNoOp:^gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/BiasAddGrad3^gradients/rnn/basic_lstm_cell/split_10_grad/concat
Р
Fgradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/split_10_grad/concat?^gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/split_10_grad/concat*'
_output_shapes
:џџџџџџџџџP
У
Hgradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/control_dependency_1Identity9gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/BiasAddGrad?^gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/group_deps*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_10_grad/BiasAddGrad*
T0*
_output_shapes
:P
ў
3gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMulMatMulFgradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ѕ
5gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_10Fgradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
Г
=gradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/group_depsNoOp4^gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul6^gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul_1
Р
Egradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/control_dependencyIdentity3gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul>^gradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Н
Ggradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/control_dependency_1Identity5gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul_1>^gradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/group_deps*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_10_grad/MatMul_1*
T0*
_output_shapes

:P
s
1gradients/rnn/basic_lstm_cell/concat_10_grad/RankConst*
dtype0*
value	B :*
_output_shapes
: 
Д
0gradients/rnn/basic_lstm_cell/concat_10_grad/modFloorMod"rnn/basic_lstm_cell/concat_10/axis1gradients/rnn/basic_lstm_cell/concat_10_grad/Rank*
T0*
_output_shapes
: 
z
2gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeShapesplit:10*
T0*
out_type0*
_output_shapes
:
Ї
3gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeNShapeNsplit:10rnn/basic_lstm_cell/Mul_29*
T0*
out_type0*
N* 
_output_shapes
::

9gradients/rnn/basic_lstm_cell/concat_10_grad/ConcatOffsetConcatOffset0gradients/rnn/basic_lstm_cell/concat_10_grad/mod3gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeN5gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeN:1*
N* 
_output_shapes
::
К
2gradients/rnn/basic_lstm_cell/concat_10_grad/SliceSliceEgradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/concat_10_grad/ConcatOffset3gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Р
4gradients/rnn/basic_lstm_cell/concat_10_grad/Slice_1SliceEgradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/control_dependency;gradients/rnn/basic_lstm_cell/concat_10_grad/ConcatOffset:15gradients/rnn/basic_lstm_cell/concat_10_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Б
=gradients/rnn/basic_lstm_cell/concat_10_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/concat_10_grad/Slice5^gradients/rnn/basic_lstm_cell/concat_10_grad/Slice_1
О
Egradients/rnn/basic_lstm_cell/concat_10_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/concat_10_grad/Slice>^gradients/rnn/basic_lstm_cell/concat_10_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/concat_10_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Ф
Ggradients/rnn/basic_lstm_cell/concat_10_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/concat_10_grad/Slice_1>^gradients/rnn/basic_lstm_cell/concat_10_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/concat_10_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_29_grad/ShapeShapernn/basic_lstm_cell/Tanh_19*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_29_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_29*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_29_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_29_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_29_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
-gradients/rnn/basic_lstm_cell/Mul_29_grad/MulMulGgradients/rnn/basic_lstm_cell/concat_10_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_29*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_29_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_29_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_29_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_29_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_29_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_29_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_29_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_19Ggradients/rnn/basic_lstm_cell/concat_10_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_29_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_29_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_29_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_29_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_29_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_19_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_19Bgradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_29_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_29Dgradients/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_5AddNBgradients/rnn/basic_lstm_cell/Mul_30_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_19_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_30_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_19_grad/ShapeShapernn/basic_lstm_cell/Mul_27*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_19_grad/Shape_1Shapernn/basic_lstm_cell/Mul_28*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_19_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_19_grad/Shape1gradients/rnn/basic_lstm_cell/Add_19_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_19_grad/SumSumgradients/AddN_5?gradients/rnn/basic_lstm_cell/Add_19_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_19_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_19_grad/Sum/gradients/rnn/basic_lstm_cell/Add_19_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_19_grad/Sum_1Sumgradients/AddN_5Agradients/rnn/basic_lstm_cell/Add_19_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_19_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_19_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_19_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_19_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_19_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_19_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_27_grad/ShapeShapernn/basic_lstm_cell/Add_17*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_27_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_27*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_27_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_27_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_27_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_27_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_27*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_27_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_27_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_27_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_27_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_27_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_27_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_27_grad/Mul_1Mulrnn/basic_lstm_cell/Add_17Bgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_27_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_27_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_27_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_27_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_27_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_28_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_28*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_28_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_18*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_28_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_28_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_28_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_28_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_18*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_28_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_28_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_28_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_28_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_28_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_28_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_28_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_28Dgradients/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_28_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_28_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_28_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_28_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_28_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_27_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_27Dgradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_28_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_28Bgradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_18_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_18Dgradients/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_18_grad/ShapeShapernn/basic_lstm_cell/split_9:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_18_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_18_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_18_grad/Shape1gradients/rnn/basic_lstm_cell/Add_18_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_18_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_27_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_18_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_18_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_18_grad/Sum/gradients/rnn/basic_lstm_cell/Add_18_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_18_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_27_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_18_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_18_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_18_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_18_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_18_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_18_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_18_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_18_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_18_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_9_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_28_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_18_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_18_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_29_grad/SigmoidGradrnn/basic_lstm_cell/Const_27*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_9_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_9_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_9_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_9_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_9Egradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/group_deps*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_9_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_9_grad/modFloorMod!rnn/basic_lstm_cell/concat_9/axis0gradients/rnn/basic_lstm_cell/concat_9_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeShapesplit:9*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeNShapeNsplit:9rnn/basic_lstm_cell/Mul_26*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_9_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_9_grad/mod2gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_9_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_9_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_9_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_9_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_9_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_9_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_9_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_9_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_9_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_9_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_9_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_9_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_9_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_9_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_26_grad/ShapeShapernn/basic_lstm_cell/Tanh_17*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_26_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_26*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_26_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_26_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_26_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_26_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_26*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_26_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_26_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_26_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_26_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_26_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_26_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_26_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_17Fgradients/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_26_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_26_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_26_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_26_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_26_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_17_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_17Bgradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_26_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_26Dgradients/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_6AddNBgradients/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_17_grad/TanhGrad*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_27_grad/Reshape*
N*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_17_grad/ShapeShapernn/basic_lstm_cell/Mul_24*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_17_grad/Shape_1Shapernn/basic_lstm_cell/Mul_25*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_17_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_17_grad/Shape1gradients/rnn/basic_lstm_cell/Add_17_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_17_grad/SumSumgradients/AddN_6?gradients/rnn/basic_lstm_cell/Add_17_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_17_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_17_grad/Sum/gradients/rnn/basic_lstm_cell/Add_17_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_17_grad/Sum_1Sumgradients/AddN_6Agradients/rnn/basic_lstm_cell/Add_17_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_17_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_17_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_17_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_17_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_17_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_17_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_24_grad/ShapeShapernn/basic_lstm_cell/Add_15*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_24_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_24*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_24_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_24_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_24_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_24_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_24*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_24_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_24_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_24_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_24_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_24_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_24_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_24_grad/Mul_1Mulrnn/basic_lstm_cell/Add_15Bgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_24_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_24_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_24_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_24_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_24_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_25_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_25*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_25_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_16*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_25_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_25_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_25_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_25_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_16*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_25_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_25_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_25_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_25_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_25_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_25_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_25_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_25Dgradients/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_25_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_25_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_25_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_25_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_25_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_24_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_24Dgradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_25_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_25Bgradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_16_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_16Dgradients/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_16_grad/ShapeShapernn/basic_lstm_cell/split_8:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_16_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_16_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_16_grad/Shape1gradients/rnn/basic_lstm_cell/Add_16_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_16_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_24_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_16_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_16_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_16_grad/Sum/gradients/rnn/basic_lstm_cell/Add_16_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_16_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_24_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_16_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_16_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_16_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_16_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_16_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_16_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_16_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_16_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_16_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_8_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_25_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_16_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_16_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_26_grad/SigmoidGradrnn/basic_lstm_cell/Const_24*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_8_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_8_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_8_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_8_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_8Egradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_8_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_8_grad/modFloorMod!rnn/basic_lstm_cell/concat_8/axis0gradients/rnn/basic_lstm_cell/concat_8_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeShapesplit:8*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeNShapeNsplit:8rnn/basic_lstm_cell/Mul_23*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_8_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_8_grad/mod2gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_8_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_8_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_8_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_8_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_8_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_8_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_8_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_8_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_8_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_8_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_8_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_8_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_8_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_8_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_23_grad/ShapeShapernn/basic_lstm_cell/Tanh_15*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_23_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_23*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_23_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_23_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_23_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_23_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_23*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_23_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_23_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_23_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_23_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_23_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_23_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_23_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_15Fgradients/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_23_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_23_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_23_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_23_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_23_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_15_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_15Bgradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_23_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_23Dgradients/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_7AddNBgradients/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_15_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_24_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_15_grad/ShapeShapernn/basic_lstm_cell/Mul_21*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_15_grad/Shape_1Shapernn/basic_lstm_cell/Mul_22*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_15_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_15_grad/Shape1gradients/rnn/basic_lstm_cell/Add_15_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_15_grad/SumSumgradients/AddN_7?gradients/rnn/basic_lstm_cell/Add_15_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_15_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_15_grad/Sum/gradients/rnn/basic_lstm_cell/Add_15_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_15_grad/Sum_1Sumgradients/AddN_7Agradients/rnn/basic_lstm_cell/Add_15_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_15_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_15_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_15_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_15_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_15_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_15_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_21_grad/ShapeShapernn/basic_lstm_cell/Add_13*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_21_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_21*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_21_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_21_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_21_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_21_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_21*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_21_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_21_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_21_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_21_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_21_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_21_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_21_grad/Mul_1Mulrnn/basic_lstm_cell/Add_13Bgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_21_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_21_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_21_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_21_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_21_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_22_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_22*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_22_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_14*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_22_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_22_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_22_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_22_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_14*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_22_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_22_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_22_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_22_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_22_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_22_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_22_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_22Dgradients/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_22_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_22_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_22_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_22_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_22_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_21_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_21Dgradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_22_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_22Bgradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_14_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_14Dgradients/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_14_grad/ShapeShapernn/basic_lstm_cell/split_7:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_14_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_14_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_14_grad/Shape1gradients/rnn/basic_lstm_cell/Add_14_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_14_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_21_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_14_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_14_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_14_grad/Sum/gradients/rnn/basic_lstm_cell/Add_14_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_14_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_21_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_14_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_14_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_14_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_14_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_14_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_14_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_14_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_14_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_14_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_7_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_22_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_14_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_14_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_23_grad/SigmoidGradrnn/basic_lstm_cell/Const_21*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_7_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_7_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_7_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_7_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_7Egradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_7_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_7_grad/modFloorMod!rnn/basic_lstm_cell/concat_7/axis0gradients/rnn/basic_lstm_cell/concat_7_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeShapesplit:7*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeNShapeNsplit:7rnn/basic_lstm_cell/Mul_20*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_7_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_7_grad/mod2gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_7_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_7_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_7_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_7_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_7_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_7_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_7_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_7_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_7_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_7_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_7_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_7_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_7_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_7_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_20_grad/ShapeShapernn/basic_lstm_cell/Tanh_13*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_20_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_20*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_20_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_20_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_20_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_20_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_20*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_20_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_20_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_20_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_20_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_20_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_20_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_20_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_13Fgradients/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_20_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_20_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_20_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_20_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_20_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_13_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_13Bgradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_20_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_20Dgradients/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_8AddNBgradients/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_13_grad/TanhGrad*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_21_grad/Reshape*
N*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_13_grad/ShapeShapernn/basic_lstm_cell/Mul_18*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_13_grad/Shape_1Shapernn/basic_lstm_cell/Mul_19*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_13_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_13_grad/Shape1gradients/rnn/basic_lstm_cell/Add_13_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_13_grad/SumSumgradients/AddN_8?gradients/rnn/basic_lstm_cell/Add_13_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_13_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_13_grad/Sum/gradients/rnn/basic_lstm_cell/Add_13_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_13_grad/Sum_1Sumgradients/AddN_8Agradients/rnn/basic_lstm_cell/Add_13_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_13_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_13_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_13_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_13_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_13_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_13_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_18_grad/ShapeShapernn/basic_lstm_cell/Add_11*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_18_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_18*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_18_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_18_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_18_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_18_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_18*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_18_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_18_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_18_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_18_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_18_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_18_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
/gradients/rnn/basic_lstm_cell/Mul_18_grad/Mul_1Mulrnn/basic_lstm_cell/Add_11Bgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_18_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_18_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_18_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_18_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_18_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_19_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_19*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_19_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_12*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_19_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_19_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_19_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_19_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_12*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_19_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_19_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_19_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_19_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_19_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_19_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_19_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_19Dgradients/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_19_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_19_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_19_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_19_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_19_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_18_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_18Dgradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_19_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_19Bgradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_12_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_12Dgradients/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_12_grad/ShapeShapernn/basic_lstm_cell/split_6:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_12_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_12_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_12_grad/Shape1gradients/rnn/basic_lstm_cell/Add_12_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_12_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_18_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_12_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_12_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_12_grad/Sum/gradients/rnn/basic_lstm_cell/Add_12_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_12_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_18_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_12_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_12_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_12_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_12_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_12_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_12_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_12_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_12_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_12_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_6_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_19_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_12_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_12_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_20_grad/SigmoidGradrnn/basic_lstm_cell/Const_18*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_6_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_6_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_6_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_6_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/group_deps*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGrad*
T0*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_6Egradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_6_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_6_grad/modFloorMod!rnn/basic_lstm_cell/concat_6/axis0gradients/rnn/basic_lstm_cell/concat_6_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeShapesplit:6*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeNShapeNsplit:6rnn/basic_lstm_cell/Mul_17*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_6_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_6_grad/mod2gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_6_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_6_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_6_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_6_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_6_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_6_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_6_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_6_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_6_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_6_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_6_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_6_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_6_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_6_grad/Slice_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_17_grad/ShapeShapernn/basic_lstm_cell/Tanh_11*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_17_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_17*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_17_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_17_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_17_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_17_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_17*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_17_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_17_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_17_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_17_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_17_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_17_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_17_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_11Fgradients/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_17_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_17_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_17_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_17_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_17_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
в
3gradients/rnn/basic_lstm_cell/Tanh_11_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_11Bgradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_17_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_17Dgradients/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_9AddNBgradients/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependency3gradients/rnn/basic_lstm_cell/Tanh_11_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_18_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_11_grad/ShapeShapernn/basic_lstm_cell/Mul_15*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Add_11_grad/Shape_1Shapernn/basic_lstm_cell/Mul_16*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Add_11_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_11_grad/Shape1gradients/rnn/basic_lstm_cell/Add_11_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Add_11_grad/SumSumgradients/AddN_9?gradients/rnn/basic_lstm_cell/Add_11_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_11_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_11_grad/Sum/gradients/rnn/basic_lstm_cell/Add_11_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
/gradients/rnn/basic_lstm_cell/Add_11_grad/Sum_1Sumgradients/AddN_9Agradients/rnn/basic_lstm_cell/Add_11_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_11_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_11_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Add_11_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_11_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_11_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_11_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_15_grad/ShapeShapernn/basic_lstm_cell/Add_9*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_15_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_15*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_15_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_15_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_15_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ъ
-gradients/rnn/basic_lstm_cell/Mul_15_grad/MulMulBgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_15*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_15_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_15_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_15_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_15_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_15_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_15_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ч
/gradients/rnn/basic_lstm_cell/Mul_15_grad/Mul_1Mulrnn/basic_lstm_cell/Add_9Bgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_15_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_15_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_15_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_15_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_15_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_16_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_16*
out_type0*
T0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_16_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_10*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_16_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_16_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_16_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_16_grad/MulMulDgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_10*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_16_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_16_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_16_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_16_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_16_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_16_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ю
/gradients/rnn/basic_lstm_cell/Mul_16_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_16Dgradients/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_16_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_16_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_16_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_16_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_16_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_15_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_15Dgradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_16_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_16Bgradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
д
3gradients/rnn/basic_lstm_cell/Tanh_10_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_10Dgradients/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Add_10_grad/ShapeShapernn/basic_lstm_cell/split_5:2*
T0*
out_type0*
_output_shapes
:
t
1gradients/rnn/basic_lstm_cell/Add_10_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
љ
?gradients/rnn/basic_lstm_cell/Add_10_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Add_10_grad/Shape1gradients/rnn/basic_lstm_cell/Add_10_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
№
-gradients/rnn/basic_lstm_cell/Add_10_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_15_grad/SigmoidGrad?gradients/rnn/basic_lstm_cell/Add_10_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Add_10_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Add_10_grad/Sum/gradients/rnn/basic_lstm_cell/Add_10_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
є
/gradients/rnn/basic_lstm_cell/Add_10_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_15_grad/SigmoidGradAgradients/rnn/basic_lstm_cell/Add_10_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
б
3gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Add_10_grad/Sum_11gradients/rnn/basic_lstm_cell/Add_10_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
:gradients/rnn/basic_lstm_cell/Add_10_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape4^gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Add_10_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape;^gradients/rnn/basic_lstm_cell/Add_10_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ћ
Dgradients/rnn/basic_lstm_cell/Add_10_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Add_10_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Add_10_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_5_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_16_grad/SigmoidGrad3gradients/rnn/basic_lstm_cell/Tanh_10_grad/TanhGradBgradients/rnn/basic_lstm_cell/Add_10_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_17_grad/SigmoidGradrnn/basic_lstm_cell/Const_15*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_5_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_5_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_5_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_5_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_5Egradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_5_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_5_grad/modFloorMod!rnn/basic_lstm_cell/concat_5/axis0gradients/rnn/basic_lstm_cell/concat_5_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeShapesplit:5*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeNShapeNsplit:5rnn/basic_lstm_cell/Mul_14*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_5_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_5_grad/mod2gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_5_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_5_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_5_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_5_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_5_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_5_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_5_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_5_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_5_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_5_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_5_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_5_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_5_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_5_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_14_grad/ShapeShapernn/basic_lstm_cell/Tanh_9*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_14_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_14*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_14_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_14_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_14_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_14_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_14*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_14_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_14_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_14_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_14_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_14_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_14_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ь
/gradients/rnn/basic_lstm_cell/Mul_14_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_9Fgradients/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_14_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_14_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_14_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_14_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_14_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
а
2gradients/rnn/basic_lstm_cell/Tanh_9_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_9Bgradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_14_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_14Dgradients/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_10AddNBgradients/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependency2gradients/rnn/basic_lstm_cell/Tanh_9_grad/TanhGrad*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_15_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_9_grad/ShapeShapernn/basic_lstm_cell/Mul_12*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Add_9_grad/Shape_1Shapernn/basic_lstm_cell/Mul_13*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Add_9_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_9_grad/Shape0gradients/rnn/basic_lstm_cell/Add_9_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Add_9_grad/SumSumgradients/AddN_10>gradients/rnn/basic_lstm_cell/Add_9_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_9_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_9_grad/Sum.gradients/rnn/basic_lstm_cell/Add_9_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
.gradients/rnn/basic_lstm_cell/Add_9_grad/Sum_1Sumgradients/AddN_10@gradients/rnn/basic_lstm_cell/Add_9_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_9_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_9_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Add_9_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_9_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_9_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_9_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_12_grad/ShapeShapernn/basic_lstm_cell/Add_7*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_12_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_12*
out_type0*
T0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_12_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_12_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_12_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
-gradients/rnn/basic_lstm_cell/Mul_12_grad/MulMulAgradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_12*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_12_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_12_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_12_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_12_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_12_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_12_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ц
/gradients/rnn/basic_lstm_cell/Mul_12_grad/Mul_1Mulrnn/basic_lstm_cell/Add_7Agradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_12_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_12_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_12_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_12_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_12_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_13_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_13*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_13_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_8*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_13_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_13_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_13_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Mul_13_grad/MulMulCgradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_8*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_13_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_13_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_13_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_13_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_13_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_13_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_13_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_13Cgradients/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_13_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_13_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_13_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_13_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_13_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_12_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_12Dgradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_13_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_13Bgradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
в
2gradients/rnn/basic_lstm_cell/Tanh_8_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_8Dgradients/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_8_grad/ShapeShapernn/basic_lstm_cell/split_4:2*
T0*
out_type0*
_output_shapes
:
s
0gradients/rnn/basic_lstm_cell/Add_8_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
і
>gradients/rnn/basic_lstm_cell/Add_8_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_8_grad/Shape0gradients/rnn/basic_lstm_cell/Add_8_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ю
,gradients/rnn/basic_lstm_cell/Add_8_grad/SumSum9gradients/rnn/basic_lstm_cell/Sigmoid_12_grad/SigmoidGrad>gradients/rnn/basic_lstm_cell/Add_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_8_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_8_grad/Sum.gradients/rnn/basic_lstm_cell/Add_8_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ђ
.gradients/rnn/basic_lstm_cell/Add_8_grad/Sum_1Sum9gradients/rnn/basic_lstm_cell/Sigmoid_12_grad/SigmoidGrad@gradients/rnn/basic_lstm_cell/Add_8_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ю
2gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_8_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Љ
9gradients/rnn/basic_lstm_cell/Add_8_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_8_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_8_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ї
Cgradients/rnn/basic_lstm_cell/Add_8_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_8_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_8_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_4_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_13_grad/SigmoidGrad2gradients/rnn/basic_lstm_cell/Tanh_8_grad/TanhGradAgradients/rnn/basic_lstm_cell/Add_8_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_14_grad/SigmoidGradrnn/basic_lstm_cell/Const_12*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_4_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_4_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_4_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_4_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_4Egradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_4_grad/RankConst*
dtype0*
value	B :*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_4_grad/modFloorMod!rnn/basic_lstm_cell/concat_4/axis0gradients/rnn/basic_lstm_cell/concat_4_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeShapesplit:4*
T0*
out_type0*
_output_shapes
:
Ѕ
2gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeNShapeNsplit:4rnn/basic_lstm_cell/Mul_11*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_4_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_4_grad/mod2gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_4_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_4_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_4_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_4_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_4_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_4_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_4_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_4_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_4_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_4_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_4_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_4_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_4_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_4_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_11_grad/ShapeShapernn/basic_lstm_cell/Tanh_7*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_11_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_11*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_11_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_11_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_11_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
-gradients/rnn/basic_lstm_cell/Mul_11_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_11*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_11_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_11_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_11_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_11_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_11_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_11_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ь
/gradients/rnn/basic_lstm_cell/Mul_11_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_7Fgradients/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_11_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_11_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_11_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_11_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_11_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ
а
2gradients/rnn/basic_lstm_cell/Tanh_7_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_7Bgradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
р
9gradients/rnn/basic_lstm_cell/Sigmoid_11_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_11Dgradients/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_11AddNBgradients/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependency2gradients/rnn/basic_lstm_cell/Tanh_7_grad/TanhGrad*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_12_grad/Reshape*
N*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_7_grad/ShapeShapernn/basic_lstm_cell/Mul_9*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Add_7_grad/Shape_1Shapernn/basic_lstm_cell/Mul_10*
out_type0*
T0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Add_7_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_7_grad/Shape0gradients/rnn/basic_lstm_cell/Add_7_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Add_7_grad/SumSumgradients/AddN_11>gradients/rnn/basic_lstm_cell/Add_7_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_7_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_7_grad/Sum.gradients/rnn/basic_lstm_cell/Add_7_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
.gradients/rnn/basic_lstm_cell/Add_7_grad/Sum_1Sumgradients/AddN_11@gradients/rnn/basic_lstm_cell/Add_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_7_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_7_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Add_7_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_7_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_7_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_7_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_9_grad/ShapeShapernn/basic_lstm_cell/Add_5*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_9_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_9*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_9_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_9_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_9_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
,gradients/rnn/basic_lstm_cell/Mul_9_grad/MulMulAgradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_9*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_9_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_9_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_9_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_9_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_9_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_9_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Х
.gradients/rnn/basic_lstm_cell/Mul_9_grad/Mul_1Mulrnn/basic_lstm_cell/Add_5Agradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_9_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_9_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_9_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_9_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_9_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/group_deps*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/rnn/basic_lstm_cell/Mul_10_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_10*
T0*
out_type0*
_output_shapes
:

1gradients/rnn/basic_lstm_cell/Mul_10_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_6*
T0*
out_type0*
_output_shapes
:
љ
?gradients/rnn/basic_lstm_cell/Mul_10_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/rnn/basic_lstm_cell/Mul_10_grad/Shape1gradients/rnn/basic_lstm_cell/Mul_10_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
-gradients/rnn/basic_lstm_cell/Mul_10_grad/MulMulCgradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_6*
T0*'
_output_shapes
:џџџџџџџџџ
ф
-gradients/rnn/basic_lstm_cell/Mul_10_grad/SumSum-gradients/rnn/basic_lstm_cell/Mul_10_grad/Mul?gradients/rnn/basic_lstm_cell/Mul_10_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
1gradients/rnn/basic_lstm_cell/Mul_10_grad/ReshapeReshape-gradients/rnn/basic_lstm_cell/Mul_10_grad/Sum/gradients/rnn/basic_lstm_cell/Mul_10_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Э
/gradients/rnn/basic_lstm_cell/Mul_10_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_10Cgradients/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
/gradients/rnn/basic_lstm_cell/Mul_10_grad/Sum_1Sum/gradients/rnn/basic_lstm_cell/Mul_10_grad/Mul_1Agradients/rnn/basic_lstm_cell/Mul_10_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
т
3gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1Reshape/gradients/rnn/basic_lstm_cell/Mul_10_grad/Sum_11gradients/rnn/basic_lstm_cell/Mul_10_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ќ
:gradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape4^gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1
Ж
Bgradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape;^gradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
М
Dgradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1;^gradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_9_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_9Cgradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
о
9gradients/rnn/basic_lstm_cell/Sigmoid_10_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_10Bgradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
в
2gradients/rnn/basic_lstm_cell/Tanh_6_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_6Dgradients/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_6_grad/ShapeShapernn/basic_lstm_cell/split_3:2*
T0*
out_type0*
_output_shapes
:
s
0gradients/rnn/basic_lstm_cell/Add_6_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
і
>gradients/rnn/basic_lstm_cell/Add_6_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_6_grad/Shape0gradients/rnn/basic_lstm_cell/Add_6_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
э
,gradients/rnn/basic_lstm_cell/Add_6_grad/SumSum8gradients/rnn/basic_lstm_cell/Sigmoid_9_grad/SigmoidGrad>gradients/rnn/basic_lstm_cell/Add_6_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_6_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_6_grad/Sum.gradients/rnn/basic_lstm_cell/Add_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ё
.gradients/rnn/basic_lstm_cell/Add_6_grad/Sum_1Sum8gradients/rnn/basic_lstm_cell/Sigmoid_9_grad/SigmoidGrad@gradients/rnn/basic_lstm_cell/Add_6_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Ю
2gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_6_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_6_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
Љ
9gradients/rnn/basic_lstm_cell/Add_6_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_6_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_6_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ї
Cgradients/rnn/basic_lstm_cell/Add_6_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_6_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_6_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_3_grad/concatConcatV29gradients/rnn/basic_lstm_cell/Sigmoid_10_grad/SigmoidGrad2gradients/rnn/basic_lstm_cell/Tanh_6_grad/TanhGradAgradients/rnn/basic_lstm_cell/Add_6_grad/tuple/control_dependency9gradients/rnn/basic_lstm_cell/Sigmoid_11_grad/SigmoidGradrnn/basic_lstm_cell/Const_9*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_3_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_3_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_3_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/group_deps*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_3_grad/concat*
T0*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_3Egradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_3_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_3_grad/modFloorMod!rnn/basic_lstm_cell/concat_3/axis0gradients/rnn/basic_lstm_cell/concat_3_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeShapesplit:3*
T0*
out_type0*
_output_shapes
:
Є
2gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeNShapeNsplit:3rnn/basic_lstm_cell/Mul_8*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_3_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_3_grad/mod2gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_3_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_3_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_3_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_3_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_3_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_3_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_3_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_3_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_3_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_3_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_3_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_3_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_3_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_8_grad/ShapeShapernn/basic_lstm_cell/Tanh_5*
out_type0*
T0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_8_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_8*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_8_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_8_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_8_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ь
,gradients/rnn/basic_lstm_cell/Mul_8_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_8*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_8_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_8_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_8_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_8_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_8_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_8_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_5Fgradients/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_8_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_8_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_8_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_8_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_8_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Я
2gradients/rnn/basic_lstm_cell/Tanh_5_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_5Agradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_8_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_8Cgradients/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_12AddNAgradients/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependency2gradients/rnn/basic_lstm_cell/Tanh_5_grad/TanhGrad*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_9_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_5_grad/ShapeShapernn/basic_lstm_cell/Mul_6*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Add_5_grad/Shape_1Shapernn/basic_lstm_cell/Mul_7*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Add_5_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_5_grad/Shape0gradients/rnn/basic_lstm_cell/Add_5_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Add_5_grad/SumSumgradients/AddN_12>gradients/rnn/basic_lstm_cell/Add_5_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_5_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_5_grad/Sum.gradients/rnn/basic_lstm_cell/Add_5_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
.gradients/rnn/basic_lstm_cell/Add_5_grad/Sum_1Sumgradients/AddN_12@gradients/rnn/basic_lstm_cell/Add_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_5_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_5_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Add_5_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_5_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_5_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_5_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_6_grad/ShapeShapernn/basic_lstm_cell/Add_3*
out_type0*
T0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_6_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_6*
out_type0*
T0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_6_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_6_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_6_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
,gradients/rnn/basic_lstm_cell/Mul_6_grad/MulMulAgradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_6*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_6_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_6_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_6_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_6_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_6_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Х
.gradients/rnn/basic_lstm_cell/Mul_6_grad/Mul_1Mulrnn/basic_lstm_cell/Add_3Agradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_6_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_6_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_6_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_6_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_6_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/group_deps*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_7_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_7*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_7_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_4*
out_type0*
T0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_7_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_7_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_7_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Mul_7_grad/MulMulCgradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_4*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_7_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_7_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_7_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_7_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_7_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_7_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_7_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_7Cgradients/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_7_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_7_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_7_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_7_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_6_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_6Cgradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
л
8gradients/rnn/basic_lstm_cell/Sigmoid_7_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_7Agradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
б
2gradients/rnn/basic_lstm_cell/Tanh_4_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_4Cgradients/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_4_grad/ShapeShapernn/basic_lstm_cell/split_2:2*
T0*
out_type0*
_output_shapes
:
s
0gradients/rnn/basic_lstm_cell/Add_4_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
і
>gradients/rnn/basic_lstm_cell/Add_4_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_4_grad/Shape0gradients/rnn/basic_lstm_cell/Add_4_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
э
,gradients/rnn/basic_lstm_cell/Add_4_grad/SumSum8gradients/rnn/basic_lstm_cell/Sigmoid_6_grad/SigmoidGrad>gradients/rnn/basic_lstm_cell/Add_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_4_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_4_grad/Sum.gradients/rnn/basic_lstm_cell/Add_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ё
.gradients/rnn/basic_lstm_cell/Add_4_grad/Sum_1Sum8gradients/rnn/basic_lstm_cell/Sigmoid_6_grad/SigmoidGrad@gradients/rnn/basic_lstm_cell/Add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
Ю
2gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_4_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Љ
9gradients/rnn/basic_lstm_cell/Add_4_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_4_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_4_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ї
Cgradients/rnn/basic_lstm_cell/Add_4_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_4_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_2_grad/concatConcatV28gradients/rnn/basic_lstm_cell/Sigmoid_7_grad/SigmoidGrad2gradients/rnn/basic_lstm_cell/Tanh_4_grad/TanhGradAgradients/rnn/basic_lstm_cell/Add_4_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/Sigmoid_8_grad/SigmoidGradrnn/basic_lstm_cell/Const_6*

Tidx0*
T0*
N*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_2_grad/concat*
T0*
data_formatNHWC*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_2_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_2_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_2_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGrad*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_2Egradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_2_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_2_grad/modFloorMod!rnn/basic_lstm_cell/concat_2/axis0gradients/rnn/basic_lstm_cell/concat_2_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeShapesplit:2*
T0*
out_type0*
_output_shapes
:
Є
2gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeNShapeNsplit:2rnn/basic_lstm_cell/Mul_5*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_2_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_2_grad/mod2gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_2_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_2_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_2_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_2_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_2_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_2_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_2_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_2_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_2_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_2_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_2_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_2_grad/tuple/group_deps*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_2_grad/Slice_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_5_grad/ShapeShapernn/basic_lstm_cell/Tanh_3*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_5_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_5*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_5_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_5_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_5_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ь
,gradients/rnn/basic_lstm_cell/Mul_5_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_5*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_5_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_5_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_5_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_5_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_5_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_5_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_3Fgradients/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_5_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_5_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_5_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_5_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_5_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Я
2gradients/rnn/basic_lstm_cell/Tanh_3_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_3Agradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_5_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_5Cgradients/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_13AddNAgradients/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependency2gradients/rnn/basic_lstm_cell/Tanh_3_grad/TanhGrad*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_6_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_3_grad/ShapeShapernn/basic_lstm_cell/Mul_3*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Add_3_grad/Shape_1Shapernn/basic_lstm_cell/Mul_4*
out_type0*
T0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_3_grad/Shape0gradients/rnn/basic_lstm_cell/Add_3_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Add_3_grad/SumSumgradients/AddN_13>gradients/rnn/basic_lstm_cell/Add_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_3_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_3_grad/Sum.gradients/rnn/basic_lstm_cell/Add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
.gradients/rnn/basic_lstm_cell/Add_3_grad/Sum_1Sumgradients/AddN_13@gradients/rnn/basic_lstm_cell/Add_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_3_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_3_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Add_3_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_3_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_3_grad/ShapeShapernn/basic_lstm_cell/Add_1*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_3_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_3*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_3_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_3_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
,gradients/rnn/basic_lstm_cell/Mul_3_grad/MulMulAgradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid_3*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_3_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_3_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_3_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_3_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Х
.gradients/rnn/basic_lstm_cell/Mul_3_grad/Mul_1Mulrnn/basic_lstm_cell/Add_1Agradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_3_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_3_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_3_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_3_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_4_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_4*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_4_grad/Shape_1Shapernn/basic_lstm_cell/Tanh_2*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_4_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_4_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Mul_4_grad/MulMulCgradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh_2*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_4_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_4_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_4_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_4_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_4_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_4Cgradients/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_4_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_4_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_4_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_4_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_3Cgradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
л
8gradients/rnn/basic_lstm_cell/Sigmoid_4_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_4Agradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
б
2gradients/rnn/basic_lstm_cell/Tanh_2_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_2Cgradients/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_2_grad/ShapeShapernn/basic_lstm_cell/split_1:2*
T0*
out_type0*
_output_shapes
:
s
0gradients/rnn/basic_lstm_cell/Add_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
і
>gradients/rnn/basic_lstm_cell/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_2_grad/Shape0gradients/rnn/basic_lstm_cell/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
э
,gradients/rnn/basic_lstm_cell/Add_2_grad/SumSum8gradients/rnn/basic_lstm_cell/Sigmoid_3_grad/SigmoidGrad>gradients/rnn/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_2_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_2_grad/Sum.gradients/rnn/basic_lstm_cell/Add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ё
.gradients/rnn/basic_lstm_cell/Add_2_grad/Sum_1Sum8gradients/rnn/basic_lstm_cell/Sigmoid_3_grad/SigmoidGrad@gradients/rnn/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ю
2gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_2_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Љ
9gradients/rnn/basic_lstm_cell/Add_2_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_2_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ї
Cgradients/rnn/basic_lstm_cell/Add_2_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_2_grad/Reshape_1*
_output_shapes
: 

1gradients/rnn/basic_lstm_cell/split_1_grad/concatConcatV28gradients/rnn/basic_lstm_cell/Sigmoid_4_grad/SigmoidGrad2gradients/rnn/basic_lstm_cell/Tanh_2_grad/TanhGradAgradients/rnn/basic_lstm_cell/Add_2_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/Sigmoid_5_grad/SigmoidGradrnn/basic_lstm_cell/Const_3*
N*

Tidx0*
T0*'
_output_shapes
:џџџџџџџџџP
Ж
8gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGradBiasAddGrad1gradients/rnn/basic_lstm_cell/split_1_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Д
=gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/group_depsNoOp9^gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad2^gradients/rnn/basic_lstm_cell/split_1_grad/concat
М
Egradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/split_1_grad/concat>^gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/split_1_grad/concat*'
_output_shapes
:џџџџџџџџџP
П
Ggradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependency_1Identity8gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad>^gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/group_deps*K
_classA
?=loc:@gradients/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad*
T0*
_output_shapes
:P
ќ
2gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMulMatMulEgradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџ
ђ
4gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1MatMulrnn/basic_lstm_cell/concat_1Egradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes

:P
А
<gradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/group_depsNoOp3^gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul5^gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1
М
Dgradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependencyIdentity2gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul=^gradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Й
Fgradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependency_1Identity4gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1=^gradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1*
_output_shapes

:P
r
0gradients/rnn/basic_lstm_cell/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Б
/gradients/rnn/basic_lstm_cell/concat_1_grad/modFloorMod!rnn/basic_lstm_cell/concat_1/axis0gradients/rnn/basic_lstm_cell/concat_1_grad/Rank*
T0*
_output_shapes
: 
x
1gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeShapesplit:1*
T0*
out_type0*
_output_shapes
:
Є
2gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeNShapeNsplit:1rnn/basic_lstm_cell/Mul_2*
T0*
out_type0*
N* 
_output_shapes
::

8gradients/rnn/basic_lstm_cell/concat_1_grad/ConcatOffsetConcatOffset/gradients/rnn/basic_lstm_cell/concat_1_grad/mod2gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeN4gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeN:1*
N* 
_output_shapes
::
Ж
1gradients/rnn/basic_lstm_cell/concat_1_grad/SliceSliceDgradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/concat_1_grad/ConcatOffset2gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeN*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
М
3gradients/rnn/basic_lstm_cell/concat_1_grad/Slice_1SliceDgradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependency:gradients/rnn/basic_lstm_cell/concat_1_grad/ConcatOffset:14gradients/rnn/basic_lstm_cell/concat_1_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Ў
<gradients/rnn/basic_lstm_cell/concat_1_grad/tuple/group_depsNoOp2^gradients/rnn/basic_lstm_cell/concat_1_grad/Slice4^gradients/rnn/basic_lstm_cell/concat_1_grad/Slice_1
К
Dgradients/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependencyIdentity1gradients/rnn/basic_lstm_cell/concat_1_grad/Slice=^gradients/rnn/basic_lstm_cell/concat_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/rnn/basic_lstm_cell/concat_1_grad/Slice*'
_output_shapes
:џџџџџџџџџ
Р
Fgradients/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependency_1Identity3gradients/rnn/basic_lstm_cell/concat_1_grad/Slice_1=^gradients/rnn/basic_lstm_cell/concat_1_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/rnn/basic_lstm_cell/concat_1_grad/Slice_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_2_grad/ShapeShapernn/basic_lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_2_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_2_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ь
,gradients/rnn/basic_lstm_cell/Mul_2_grad/MulMulFgradients/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Sigmoid_2*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_2_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_2_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_2_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_2_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_2_grad/Mul_1Mulrnn/basic_lstm_cell/Tanh_1Fgradients/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_2_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_2_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_2_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Я
2gradients/rnn/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradrnn/basic_lstm_cell/Tanh_1Agradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
н
8gradients/rnn/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_2Cgradients/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/AddN_14AddNAgradients/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependency2gradients/rnn/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_3_grad/Reshape*
N*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Add_1_grad/ShapeShapernn/basic_lstm_cell/Mul*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Add_1_grad/Shape_1Shapernn/basic_lstm_cell/Mul_1*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Add_1_grad/Shape0gradients/rnn/basic_lstm_cell/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ц
,gradients/rnn/basic_lstm_cell/Add_1_grad/SumSumgradients/AddN_14>gradients/rnn/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Add_1_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Add_1_grad/Sum.gradients/rnn/basic_lstm_cell/Add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ъ
.gradients/rnn/basic_lstm_cell/Add_1_grad/Sum_1Sumgradients/AddN_14@gradients/rnn/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Add_1_grad/Sum_10gradients/rnn/basic_lstm_cell/Add_1_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape3^gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape:^gradients/rnn/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Add_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

,gradients/rnn/basic_lstm_cell/Mul_grad/ShapeShape rnn/BasicLSTMCellZeroState/zeros*
T0*
out_type0*
_output_shapes
:

.gradients/rnn/basic_lstm_cell/Mul_grad/Shape_1Shapernn/basic_lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
№
<gradients/rnn/basic_lstm_cell/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/rnn/basic_lstm_cell/Mul_grad/Shape.gradients/rnn/basic_lstm_cell/Mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
У
*gradients/rnn/basic_lstm_cell/Mul_grad/MulMulAgradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependencyrnn/basic_lstm_cell/Sigmoid*
T0*'
_output_shapes
:џџџџџџџџџ
л
*gradients/rnn/basic_lstm_cell/Mul_grad/SumSum*gradients/rnn/basic_lstm_cell/Mul_grad/Mul<gradients/rnn/basic_lstm_cell/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
г
.gradients/rnn/basic_lstm_cell/Mul_grad/ReshapeReshape*gradients/rnn/basic_lstm_cell/Mul_grad/Sum,gradients/rnn/basic_lstm_cell/Mul_grad/Shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
,gradients/rnn/basic_lstm_cell/Mul_grad/Mul_1Mul rnn/BasicLSTMCellZeroState/zerosAgradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_grad/Sum_1Sum,gradients/rnn/basic_lstm_cell/Mul_grad/Mul_1>gradients/rnn/basic_lstm_cell/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_grad/Reshape_1Reshape,gradients/rnn/basic_lstm_cell/Mul_grad/Sum_1.gradients/rnn/basic_lstm_cell/Mul_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
7gradients/rnn/basic_lstm_cell/Mul_grad/tuple/group_depsNoOp/^gradients/rnn/basic_lstm_cell/Mul_grad/Reshape1^gradients/rnn/basic_lstm_cell/Mul_grad/Reshape_1
Њ
?gradients/rnn/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity.gradients/rnn/basic_lstm_cell/Mul_grad/Reshape8^gradients/rnn/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn/basic_lstm_cell/Mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
А
Agradients/rnn/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity0gradients/rnn/basic_lstm_cell/Mul_grad/Reshape_18^gradients/rnn/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

.gradients/rnn/basic_lstm_cell/Mul_1_grad/ShapeShapernn/basic_lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:

0gradients/rnn/basic_lstm_cell/Mul_1_grad/Shape_1Shapernn/basic_lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
і
>gradients/rnn/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/rnn/basic_lstm_cell/Mul_1_grad/Shape0gradients/rnn/basic_lstm_cell/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ф
,gradients/rnn/basic_lstm_cell/Mul_1_grad/MulMulCgradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1rnn/basic_lstm_cell/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
с
,gradients/rnn/basic_lstm_cell/Mul_1_grad/SumSum,gradients/rnn/basic_lstm_cell/Mul_1_grad/Mul>gradients/rnn/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
й
0gradients/rnn/basic_lstm_cell/Mul_1_grad/ReshapeReshape,gradients/rnn/basic_lstm_cell/Mul_1_grad/Sum.gradients/rnn/basic_lstm_cell/Mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ы
.gradients/rnn/basic_lstm_cell/Mul_1_grad/Mul_1Mulrnn/basic_lstm_cell/Sigmoid_1Cgradients/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
ч
.gradients/rnn/basic_lstm_cell/Mul_1_grad/Sum_1Sum.gradients/rnn/basic_lstm_cell/Mul_1_grad/Mul_1@gradients/rnn/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
п
2gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1Reshape.gradients/rnn/basic_lstm_cell/Mul_1_grad/Sum_10gradients/rnn/basic_lstm_cell/Mul_1_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ
Љ
9gradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape3^gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1
В
Agradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape:^gradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/group_deps*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ
И
Cgradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1:^gradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
з
6gradients/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/SigmoidAgradients/rnn/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
л
8gradients/rnn/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradrnn/basic_lstm_cell/Sigmoid_1Agradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
Э
0gradients/rnn/basic_lstm_cell/Tanh_grad/TanhGradTanhGradrnn/basic_lstm_cell/TanhCgradients/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ

,gradients/rnn/basic_lstm_cell/Add_grad/ShapeShapernn/basic_lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:
q
.gradients/rnn/basic_lstm_cell/Add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
№
<gradients/rnn/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/rnn/basic_lstm_cell/Add_grad/Shape.gradients/rnn/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ч
*gradients/rnn/basic_lstm_cell/Add_grad/SumSum6gradients/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGrad<gradients/rnn/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
г
.gradients/rnn/basic_lstm_cell/Add_grad/ReshapeReshape*gradients/rnn/basic_lstm_cell/Add_grad/Sum,gradients/rnn/basic_lstm_cell/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ы
,gradients/rnn/basic_lstm_cell/Add_grad/Sum_1Sum6gradients/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGrad>gradients/rnn/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ш
0gradients/rnn/basic_lstm_cell/Add_grad/Reshape_1Reshape,gradients/rnn/basic_lstm_cell/Add_grad/Sum_1.gradients/rnn/basic_lstm_cell/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ѓ
7gradients/rnn/basic_lstm_cell/Add_grad/tuple/group_depsNoOp/^gradients/rnn/basic_lstm_cell/Add_grad/Reshape1^gradients/rnn/basic_lstm_cell/Add_grad/Reshape_1
Њ
?gradients/rnn/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity.gradients/rnn/basic_lstm_cell/Add_grad/Reshape8^gradients/rnn/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/rnn/basic_lstm_cell/Add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

Agradients/rnn/basic_lstm_cell/Add_grad/tuple/control_dependency_1Identity0gradients/rnn/basic_lstm_cell/Add_grad/Reshape_18^gradients/rnn/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: 

/gradients/rnn/basic_lstm_cell/split_grad/concatConcatV28gradients/rnn/basic_lstm_cell/Sigmoid_1_grad/SigmoidGrad0gradients/rnn/basic_lstm_cell/Tanh_grad/TanhGrad?gradients/rnn/basic_lstm_cell/Add_grad/tuple/control_dependency8gradients/rnn/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradrnn/basic_lstm_cell/Const*
T0*
N*

Tidx0*'
_output_shapes
:џџџџџџџџџP
В
6gradients/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients/rnn/basic_lstm_cell/split_grad/concat*
data_formatNHWC*
T0*
_output_shapes
:P
Ў
;gradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOp7^gradients/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad0^gradients/rnn/basic_lstm_cell/split_grad/concat
Д
Cgradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity/gradients/rnn/basic_lstm_cell/split_grad/concat<^gradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/rnn/basic_lstm_cell/split_grad/concat*'
_output_shapes
:џџџџџџџџџP
З
Egradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad<^gradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes
:P
ј
0gradients/rnn/basic_lstm_cell/MatMul_grad/MatMulMatMulCgradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:џџџџџџџџџ
ь
2gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul_1MatMulrnn/basic_lstm_cell/concatCgradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:P
Њ
:gradients/rnn/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOp1^gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul3^gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul_1
Д
Bgradients/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentity0gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul;^gradients/rnn/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
Б
Dgradients/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1Identity2gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul_1;^gradients/rnn/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/rnn/basic_lstm_cell/MatMul_grad/MatMul_1*
_output_shapes

:P
Љ

gradients/AddN_15AddNHgradients/rnn/basic_lstm_cell/BiasAdd_15_grad/tuple/control_dependency_1Hgradients/rnn/basic_lstm_cell/BiasAdd_14_grad/tuple/control_dependency_1Hgradients/rnn/basic_lstm_cell/BiasAdd_13_grad/tuple/control_dependency_1Hgradients/rnn/basic_lstm_cell/BiasAdd_12_grad/tuple/control_dependency_1Hgradients/rnn/basic_lstm_cell/BiasAdd_11_grad/tuple/control_dependency_1Hgradients/rnn/basic_lstm_cell/BiasAdd_10_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependency_1Egradients/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients/rnn/basic_lstm_cell/BiasAdd_15_grad/BiasAddGrad*
N*
_output_shapes
:P


gradients/AddN_16AddNGgradients/rnn/basic_lstm_cell/MatMul_15_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/MatMul_14_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/MatMul_13_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/MatMul_12_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/MatMul_11_grad/tuple/control_dependency_1Ggradients/rnn/basic_lstm_cell/MatMul_10_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependency_1Fgradients/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependency_1Dgradients/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/rnn/basic_lstm_cell/MatMul_15_grad/MatMul_1*
N*
_output_shapes

:P
}
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
shape: *
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
_output_shapes
: 
­
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 
i
beta1_power/readIdentitybeta1_power*
_class
loc:@dense/bias*
T0*
_output_shapes
: 
}
beta2_power/initial_valueConst*
valueB
 *wО?*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
shape: *
_output_shapes
: 
­
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
use_locking(*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
i
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
С
Arnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   P   *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:
Ћ
7rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 

1rnn/basic_lstm_cell/kernel/Adam/Initializer/zerosFillArnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensor7rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
Т
rnn/basic_lstm_cell/kernel/Adam
VariableV2*
shape
:P*
shared_name *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
	container *
_output_shapes

:P

&rnn/basic_lstm_cell/kernel/Adam/AssignAssignrnn/basic_lstm_cell/kernel/Adam1rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:P
Љ
$rnn/basic_lstm_cell/kernel/Adam/readIdentityrnn/basic_lstm_cell/kernel/Adam*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
У
Crnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   P   *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:
­
9rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes
: 
Ѕ
3rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillCrnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensor9rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
Ф
!rnn/basic_lstm_cell/kernel/Adam_1
VariableV2*
shape
:P*
shared_name *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
	container *
_output_shapes

:P

(rnn/basic_lstm_cell/kernel/Adam_1/AssignAssign!rnn/basic_lstm_cell/kernel/Adam_13rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:P
­
&rnn/basic_lstm_cell/kernel/Adam_1/readIdentity!rnn/basic_lstm_cell/kernel/Adam_1*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
Љ
/rnn/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
valueBP*    *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
:P
Ж
rnn/basic_lstm_cell/bias/Adam
VariableV2*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
	container *
shape:P*
shared_name *
_output_shapes
:P
љ
$rnn/basic_lstm_cell/bias/Adam/AssignAssignrnn/basic_lstm_cell/bias/Adam/rnn/basic_lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:P

"rnn/basic_lstm_cell/bias/Adam/readIdentityrnn/basic_lstm_cell/bias/Adam*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes
:P
Ћ
1rnn/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
valueBP*    *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
:P
И
rnn/basic_lstm_cell/bias/Adam_1
VariableV2*
shared_name *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
	container *
shape:P*
_output_shapes
:P
џ
&rnn/basic_lstm_cell/bias/Adam_1/AssignAssignrnn/basic_lstm_cell/bias/Adam_11rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:P
Ѓ
$rnn/basic_lstm_cell/bias/Adam_1/readIdentityrnn/basic_lstm_cell/bias/Adam_1*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes
:P

#dense/kernel/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/kernel*
dtype0*
_output_shapes

:
І
dense/kernel/Adam
VariableV2*
shape
:*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container *
_output_shapes

:
Э
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:

dense/kernel/Adam/readIdentitydense/kernel/Adam*
T0*
_class
loc:@dense/kernel*
_output_shapes

:

%dense/kernel/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/kernel*
dtype0*
_output_shapes

:
Ј
dense/kernel/Adam_1
VariableV2*
_class
loc:@dense/kernel*
dtype0*
	container *
shape
:*
shared_name *
_output_shapes

:
г
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:

dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0*
_class
loc:@dense/kernel*
_output_shapes

:

!dense/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_class
loc:@dense/bias*
_output_shapes
:

dense/bias/Adam
VariableV2*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
shape:*
_output_shapes
:
С
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros*
_class
loc:@dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
u
dense/bias/Adam/readIdentitydense/bias/Adam*
T0*
_class
loc:@dense/bias*
_output_shapes
:

#dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:

dense/bias/Adam_1
VariableV2*
shape:*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container *
_output_shapes
:
Ч
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
y
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
T0*
_class
loc:@dense/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *
з#<*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
valueB
 *wО?*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 

0Adam/update_rnn/basic_lstm_cell/kernel/ApplyAdam	ApplyAdamrnn/basic_lstm_cell/kernelrnn/basic_lstm_cell/kernel/Adam!rnn/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_16*
use_locking( *
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
use_nesterov( *
_output_shapes

:P
џ
.Adam/update_rnn/basic_lstm_cell/bias/ApplyAdam	ApplyAdamrnn/basic_lstm_cell/biasrnn/basic_lstm_cell/bias/Adamrnn/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_15*
use_locking( *
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
use_nesterov( *
_output_shapes
:P
ь
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
use_nesterov( *
_output_shapes

:
п
 Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
use_nesterov( *
use_locking( *
T0*
_output_shapes
:

Adam/mulMulbeta1_power/read
Adam/beta1!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam/^Adam/update_rnn/basic_lstm_cell/bias/ApplyAdam1^Adam/update_rnn/basic_lstm_cell/kernel/ApplyAdam*
T0*
_class
loc:@dense/bias*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
_class
loc:@dense/bias*
validate_shape(*
use_locking( *
T0*
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam/^Adam/update_rnn/basic_lstm_cell/bias/ApplyAdam1^Adam/update_rnn/basic_lstm_cell/kernel/ApplyAdam*
_class
loc:@dense/bias*
T0*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
н
Adam/updateNoOp^Adam/Assign^Adam/Assign_1!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam/^Adam/update_rnn/basic_lstm_cell/bias/ApplyAdam1^Adam/update_rnn/basic_lstm_cell/kernel/ApplyAdam
z

Adam/valueConst^Adam/update*
value	B	 R*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
~
Adam	AssignAddglobal_step
Adam/value*
use_locking( *
T0	*
_class
loc:@global_step*
_output_shapes
: 
^
subSubIteratorGetNext:1dense/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
G
SquareSquaresub*
T0*'
_output_shapes
:џџџџџџџџџ
І
/root_mean_squared_error/total/Initializer/zerosConst*
valueB
 *    *0
_class&
$"loc:@root_mean_squared_error/total*
dtype0*
_output_shapes
: 
Г
root_mean_squared_error/total
VariableV2*
shared_name *0
_class&
$"loc:@root_mean_squared_error/total*
dtype0*
	container *
shape: *
_output_shapes
: 
њ
$root_mean_squared_error/total/AssignAssignroot_mean_squared_error/total/root_mean_squared_error/total/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@root_mean_squared_error/total*
validate_shape(*
_output_shapes
: 
 
"root_mean_squared_error/total/readIdentityroot_mean_squared_error/total*
T0*0
_class&
$"loc:@root_mean_squared_error/total*
_output_shapes
: 
І
/root_mean_squared_error/count/Initializer/zerosConst*
valueB
 *    *0
_class&
$"loc:@root_mean_squared_error/count*
dtype0*
_output_shapes
: 
Г
root_mean_squared_error/count
VariableV2*
shared_name *0
_class&
$"loc:@root_mean_squared_error/count*
dtype0*
	container *
shape: *
_output_shapes
: 
њ
$root_mean_squared_error/count/AssignAssignroot_mean_squared_error/count/root_mean_squared_error/count/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@root_mean_squared_error/count*
validate_shape(*
_output_shapes
: 
 
"root_mean_squared_error/count/readIdentityroot_mean_squared_error/count*
T0*0
_class&
$"loc:@root_mean_squared_error/count*
_output_shapes
: 
]
root_mean_squared_error/SizeSizeSquare*
T0*
out_type0*
_output_shapes
: 
w
!root_mean_squared_error/ToFloat_1Castroot_mean_squared_error/Size*

SrcT0*

DstT0*
_output_shapes
: 
n
root_mean_squared_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

root_mean_squared_error/SumSumSquareroot_mean_squared_error/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
а
!root_mean_squared_error/AssignAdd	AssignAddroot_mean_squared_error/totalroot_mean_squared_error/Sum*
use_locking( *
T0*0
_class&
$"loc:@root_mean_squared_error/total*
_output_shapes
: 
с
#root_mean_squared_error/AssignAdd_1	AssignAddroot_mean_squared_error/count!root_mean_squared_error/ToFloat_1^Square*0
_class&
$"loc:@root_mean_squared_error/count*
use_locking( *
T0*
_output_shapes
: 

root_mean_squared_error/truedivRealDiv"root_mean_squared_error/total/read"root_mean_squared_error/count/read*
T0*
_output_shapes
: 
g
"root_mean_squared_error/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 

root_mean_squared_error/GreaterGreater"root_mean_squared_error/count/read"root_mean_squared_error/zeros_like*
T0*
_output_shapes
: 
Ў
root_mean_squared_error/valueSelectroot_mean_squared_error/Greaterroot_mean_squared_error/truediv"root_mean_squared_error/zeros_like*
T0*
_output_shapes
: 

!root_mean_squared_error/truediv_1RealDiv!root_mean_squared_error/AssignAdd#root_mean_squared_error/AssignAdd_1*
T0*
_output_shapes
: 
i
$root_mean_squared_error/zeros_like_1Const*
dtype0*
valueB
 *    *
_output_shapes
: 

!root_mean_squared_error/Greater_1Greater#root_mean_squared_error/AssignAdd_1$root_mean_squared_error/zeros_like_1*
T0*
_output_shapes
: 
И
!root_mean_squared_error/update_opSelect!root_mean_squared_error/Greater_1!root_mean_squared_error/truediv_1$root_mean_squared_error/zeros_like_1*
T0*
_output_shapes
: 
L
SqrtSqrtroot_mean_squared_error/value*
T0*
_output_shapes
: 
R
Sqrt_1Sqrt!root_mean_squared_error/update_op*
T0*
_output_shapes
: 
`
sub_1Subdense/BiasAddIteratorGetNext:1*
T0*'
_output_shapes
:џџџџџџџџџ
C
AbsAbssub_1*
T0*'
_output_shapes
:џџџџџџџџџ

+mean_absolute_error/total/Initializer/zerosConst*
valueB
 *    *,
_class"
 loc:@mean_absolute_error/total*
dtype0*
_output_shapes
: 
Ћ
mean_absolute_error/total
VariableV2*
	container *
shape: *
shared_name *,
_class"
 loc:@mean_absolute_error/total*
dtype0*
_output_shapes
: 
ъ
 mean_absolute_error/total/AssignAssignmean_absolute_error/total+mean_absolute_error/total/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@mean_absolute_error/total*
validate_shape(*
_output_shapes
: 

mean_absolute_error/total/readIdentitymean_absolute_error/total*
T0*,
_class"
 loc:@mean_absolute_error/total*
_output_shapes
: 

+mean_absolute_error/count/Initializer/zerosConst*
dtype0*
valueB
 *    *,
_class"
 loc:@mean_absolute_error/count*
_output_shapes
: 
Ћ
mean_absolute_error/count
VariableV2*
shared_name *,
_class"
 loc:@mean_absolute_error/count*
dtype0*
	container *
shape: *
_output_shapes
: 
ъ
 mean_absolute_error/count/AssignAssignmean_absolute_error/count+mean_absolute_error/count/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@mean_absolute_error/count*
validate_shape(*
_output_shapes
: 

mean_absolute_error/count/readIdentitymean_absolute_error/count*
T0*,
_class"
 loc:@mean_absolute_error/count*
_output_shapes
: 
V
mean_absolute_error/SizeSizeAbs*
T0*
out_type0*
_output_shapes
: 
o
mean_absolute_error/ToFloat_1Castmean_absolute_error/Size*

DstT0*

SrcT0*
_output_shapes
: 
j
mean_absolute_error/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
|
mean_absolute_error/SumSumAbsmean_absolute_error/Const*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
Р
mean_absolute_error/AssignAdd	AssignAddmean_absolute_error/totalmean_absolute_error/Sum*
use_locking( *
T0*,
_class"
 loc:@mean_absolute_error/total*
_output_shapes
: 
Ю
mean_absolute_error/AssignAdd_1	AssignAddmean_absolute_error/countmean_absolute_error/ToFloat_1^Abs*
use_locking( *
T0*,
_class"
 loc:@mean_absolute_error/count*
_output_shapes
: 

mean_absolute_error/truedivRealDivmean_absolute_error/total/readmean_absolute_error/count/read*
T0*
_output_shapes
: 
c
mean_absolute_error/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 

mean_absolute_error/GreaterGreatermean_absolute_error/count/readmean_absolute_error/zeros_like*
T0*
_output_shapes
: 

mean_absolute_error/valueSelectmean_absolute_error/Greatermean_absolute_error/truedivmean_absolute_error/zeros_like*
T0*
_output_shapes
: 

mean_absolute_error/truediv_1RealDivmean_absolute_error/AssignAddmean_absolute_error/AssignAdd_1*
T0*
_output_shapes
: 
e
 mean_absolute_error/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 

mean_absolute_error/Greater_1Greatermean_absolute_error/AssignAdd_1 mean_absolute_error/zeros_like_1*
T0*
_output_shapes
: 
Ј
mean_absolute_error/update_opSelectmean_absolute_error/Greater_1mean_absolute_error/truediv_1 mean_absolute_error/zeros_like_1*
T0*
_output_shapes
: 

mean/total/Initializer/zerosConst*
valueB
 *    *
_class
loc:@mean/total*
dtype0*
_output_shapes
: 


mean/total
VariableV2*
shape: *
shared_name *
_class
loc:@mean/total*
dtype0*
	container *
_output_shapes
: 
Ў
mean/total/AssignAssign
mean/totalmean/total/Initializer/zeros*
use_locking(*
T0*
_class
loc:@mean/total*
validate_shape(*
_output_shapes
: 
g
mean/total/readIdentity
mean/total*
T0*
_class
loc:@mean/total*
_output_shapes
: 

mean/count/Initializer/zerosConst*
valueB
 *    *
_class
loc:@mean/count*
dtype0*
_output_shapes
: 


mean/count
VariableV2*
shape: *
shared_name *
_class
loc:@mean/count*
dtype0*
	container *
_output_shapes
: 
Ў
mean/count/AssignAssign
mean/countmean/count/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@mean/count*
_output_shapes
: 
g
mean/count/readIdentity
mean/count*
T0*
_class
loc:@mean/count*
_output_shapes
: 
K
	mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Q
mean/ToFloat_1Cast	mean/Size*

SrcT0*

DstT0*
_output_shapes
: 
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
s
mean/SumSummean_squared_error/value
mean/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
T0*
_class
loc:@mean/total*
use_locking( *
_output_shapes
: 
Ї
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1^mean_squared_error/value*
use_locking( *
T0*
_class
loc:@mean/count*
_output_shapes
: 
Z
mean/truedivRealDivmean/total/readmean/count/read*
T0*
_output_shapes
: 
T
mean/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
mean/GreaterGreatermean/count/readmean/zeros_like*
T0*
_output_shapes
: 
b

mean/valueSelectmean/Greatermean/truedivmean/zeros_like*
T0*
_output_shapes
: 
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
T0*
_output_shapes
: 
V
mean/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
_
mean/Greater_1Greatermean/AssignAdd_1mean/zeros_like_1*
T0*
_output_shapes
: 
l
mean/update_opSelectmean/Greater_1mean/truediv_1mean/zeros_like_1*
T0*
_output_shapes
: 
L

group_depsNoOp^Sqrt_1^mean/update_op^mean_absolute_error/update_op
{
eval_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 

	eval_step
VariableV2*
_class
loc:@eval_step*
dtype0	*
	container *
shape: *
shared_name *
_output_shapes
: 
Њ
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
validate_shape(*
use_locking(*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
d
eval_step/readIdentity	eval_step*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
Q
AssignAdd/valueConst*
dtype0	*
value	B	 R*
_output_shapes
: 

	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
use_locking(*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
U
readIdentity	eval_step
^AssignAdd^group_deps*
T0	*
_output_shapes
: 
;
IdentityIdentityread*
T0	*
_output_shapes
: 
Ы
initNoOp^beta1_power/Assign^beta2_power/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^global_step/Assign%^rnn/basic_lstm_cell/bias/Adam/Assign'^rnn/basic_lstm_cell/bias/Adam_1/Assign ^rnn/basic_lstm_cell/bias/Assign'^rnn/basic_lstm_cell/kernel/Adam/Assign)^rnn/basic_lstm_cell/kernel/Adam_1/Assign"^rnn/basic_lstm_cell/kernel/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
П
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedrnn/basic_lstm_cell/kernel*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Л
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedrnn/basic_lstm_cell/bias*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
: 
Ѓ
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
 
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedbeta1_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
 
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializedbeta2_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ф
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializedrnn/basic_lstm_cell/kernel/Adam*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Ц
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitialized!rnn/basic_lstm_cell/kernel/Adam_1*
dtype0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes
: 
Р
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedrnn/basic_lstm_cell/bias/Adam*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
: 
У
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedrnn/basic_lstm_cell/bias/Adam_1*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
: 
Љ
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitializeddense/kernel/Adam*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ћ
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitializeddense/kernel/Adam_1*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ѕ
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitializeddense/bias/Adam*
dtype0*
_class
loc:@dense/bias*
_output_shapes
: 
Ї
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitializeddense/bias/Adam_1*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ц
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializedroot_mean_squared_error/total*0
_class&
$"loc:@root_mean_squared_error/total*
dtype0*
_output_shapes
: 
Ц
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializedroot_mean_squared_error/count*0
_class&
$"loc:@root_mean_squared_error/count*
dtype0*
_output_shapes
: 
О
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitializedmean_absolute_error/total*,
_class"
 loc:@mean_absolute_error/total*
dtype0*
_output_shapes
: 
О
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitializedmean_absolute_error/count*,
_class"
 loc:@mean_absolute_error/count*
dtype0*
_output_shapes
: 
 
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized
mean/total*
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
 
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 

7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
Я

$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_21"/device:CPU:0*
N*
T0
*

axis *
_output_shapes
:

)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
л
$report_uninitialized_variables/ConstConst"/device:CPU:0*ѓ
valueщBцBglobal_stepBrnn/basic_lstm_cell/kernelBrnn/basic_lstm_cell/biasBdense/kernelB
dense/biasBbeta1_powerBbeta2_powerBrnn/basic_lstm_cell/kernel/AdamB!rnn/basic_lstm_cell/kernel/Adam_1Brnn/basic_lstm_cell/bias/AdamBrnn/basic_lstm_cell/bias/Adam_1Bdense/kernel/AdamBdense/kernel/Adam_1Bdense/bias/AdamBdense/bias/Adam_1Broot_mean_squared_error/totalBroot_mean_squared_error/countBmean_absolute_error/totalBmean_absolute_error/countB
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:

1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
dtype0*
valueB:*
_output_shapes
:
ш
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
:

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 

3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
dtype0*
valueB: *
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
№
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
№
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
Index0*
T0*
_output_shapes
: 
О
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
ї
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*

Tidx0*
T0*
N*
_output_shapes
:
к
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ъ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
Tshape0*
T0
*
_output_shapes
:
В
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:џџџџџџџџџ
Х
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*
T0	*
squeeze_dims
*#
_output_shapes
:џџџџџџџџџ

9report_uninitialized_variables/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Х
4report_uninitialized_variables/boolean_mask/GatherV2GatherV23report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze9report_uninitialized_variables/boolean_mask/GatherV2/axis"/device:CPU:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
О
concatConcatV24report_uninitialized_variables/boolean_mask/GatherV2$report_uninitialized_resources/Constconcat/axis*

Tidx0*
T0*
N*#
_output_shapes
:џџџџџџџџџ
Ё
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
С
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedrnn/basic_lstm_cell/kernel*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Н
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedrnn/basic_lstm_cell/bias*
dtype0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes
: 
Ѕ
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ё
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ђ
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializedbeta1_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ђ
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializedbeta2_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Ц
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializedrnn/basic_lstm_cell/kernel/Adam*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Ш
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitialized!rnn/basic_lstm_cell/kernel/Adam_1*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
Т
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializedrnn/basic_lstm_cell/bias/Adam*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
: 
Х
9report_uninitialized_variables_1/IsVariableInitialized_10IsVariableInitializedrnn/basic_lstm_cell/bias/Adam_1*
dtype0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes
: 
Ћ
9report_uninitialized_variables_1/IsVariableInitialized_11IsVariableInitializeddense/kernel/Adam*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
­
9report_uninitialized_variables_1/IsVariableInitialized_12IsVariableInitializeddense/kernel/Adam_1*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ї
9report_uninitialized_variables_1/IsVariableInitialized_13IsVariableInitializeddense/bias/Adam*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Љ
9report_uninitialized_variables_1/IsVariableInitialized_14IsVariableInitializeddense/bias/Adam_1*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
р
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_99report_uninitialized_variables_1/IsVariableInitialized_109report_uninitialized_variables_1/IsVariableInitialized_119report_uninitialized_variables_1/IsVariableInitialized_129report_uninitialized_variables_1/IsVariableInitialized_139report_uninitialized_variables_1/IsVariableInitialized_14"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:

+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:
Ц
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*м
valueвBЯBglobal_stepBrnn/basic_lstm_cell/kernelBrnn/basic_lstm_cell/biasBdense/kernelB
dense/biasBbeta1_powerBbeta2_powerBrnn/basic_lstm_cell/kernel/AdamB!rnn/basic_lstm_cell/kernel/Adam_1Brnn/basic_lstm_cell/bias/AdamBrnn/basic_lstm_cell/bias/Adam_1Bdense/kernel/AdamBdense/kernel/Adam_1Bdense/bias/AdamBdense/bias/Adam_1*
dtype0*
_output_shapes
:

3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
dtype0*
valueB:*
_output_shapes
:

Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
dtype0*
valueB:*
_output_shapes
:
ђ
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:

Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
dtype0*
valueB: *
_output_shapes
:

2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
dtype0*
valueB:*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
new_axis_mask *
end_mask*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
_output_shapes
: 
Т
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
N*
T0*

axis *
_output_shapes
:

9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
T0*
N*

Tidx0*
_output_shapes
:
р
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:

=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
№
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
Ж
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:џџџџџџџџџ
Щ
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:џџџџџџџџџ

;report_uninitialized_variables_1/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Э
6report_uninitialized_variables_1/boolean_mask/GatherV2GatherV25report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze;report_uninitialized_variables_1/boolean_mask/GatherV2/axis"/device:CPU:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ
н
init_2NoOp^eval_step/Assign^mean/count/Assign^mean/total/Assign!^mean_absolute_error/count/Assign!^mean_absolute_error/total/Assign%^root_mean_squared_error/count/Assign%^root_mean_squared_error/total/Assign

init_all_tablesNoOp

init_3NoOp
8
group_deps_2NoOp^init_2^init_3^init_all_tables
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_2f099ea2be324ff7a3a206e8b7b3e60a/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
И
save/SaveV2/tensor_namesConst"/device:CPU:0*м
valueвBЯBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bglobal_stepBrnn/basic_lstm_cell/biasBrnn/basic_lstm_cell/bias/AdamBrnn/basic_lstm_cell/bias/Adam_1Brnn/basic_lstm_cell/kernelBrnn/basic_lstm_cell/kernel/AdamB!rnn/basic_lstm_cell/kernel/Adam_1*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst"/device:CPU:0*1
value(B&B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
и
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1global_steprnn/basic_lstm_cell/biasrnn/basic_lstm_cell/bias/Adamrnn/basic_lstm_cell/bias/Adam_1rnn/basic_lstm_cell/kernelrnn/basic_lstm_cell/kernel/Adam!rnn/basic_lstm_cell/kernel/Adam_1"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
Л
save/RestoreV2/tensor_namesConst"/device:CPU:0*м
valueвBЯBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bglobal_stepBrnn/basic_lstm_cell/biasBrnn/basic_lstm_cell/bias/AdamBrnn/basic_lstm_cell/bias/Adam_1Brnn/basic_lstm_cell/kernelBrnn/basic_lstm_cell/kernel/AdamB!rnn/basic_lstm_cell/kernel/Adam_1*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*1
value(B&B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
х
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*P
_output_shapes>
<:::::::::::::::

save/AssignAssignbeta1_powersave/RestoreV2*
T0*
_class
loc:@dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 

save/Assign_1Assignbeta2_powersave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
: 
Ђ
save/Assign_2Assign
dense/biassave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
Ї
save/Assign_3Assigndense/bias/Adamsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
Љ
save/Assign_4Assigndense/bias/Adam_1save/RestoreV2:4*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
Њ
save/Assign_5Assigndense/kernelsave/RestoreV2:5*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
Џ
save/Assign_6Assigndense/kernel/Adamsave/RestoreV2:6*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
Б
save/Assign_7Assigndense/kernel/Adam_1save/RestoreV2:7*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes

:
 
save/Assign_8Assignglobal_stepsave/RestoreV2:8*
T0	*
_class
loc:@global_step*
validate_shape(*
use_locking(*
_output_shapes
: 
О
save/Assign_9Assignrnn/basic_lstm_cell/biassave/RestoreV2:9*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
use_locking(*
_output_shapes
:P
Х
save/Assign_10Assignrnn/basic_lstm_cell/bias/Adamsave/RestoreV2:10*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes
:P
Ч
save/Assign_11Assignrnn/basic_lstm_cell/bias/Adam_1save/RestoreV2:11*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
use_locking(*
_output_shapes
:P
Ш
save/Assign_12Assignrnn/basic_lstm_cell/kernelsave/RestoreV2:12*
validate_shape(*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes

:P
Э
save/Assign_13Assignrnn/basic_lstm_cell/kernel/Adamsave/RestoreV2:13*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:P
Я
save/Assign_14Assign!rnn/basic_lstm_cell/kernel/Adam_1save/RestoreV2:14*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(*
_output_shapes

:P

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shardХ'
Ѓ
1
_make_dataset_IBZ7uUdCWTo
repeatdatasetb
%TextLineDataset/MatchingFiles/patternConst*%
valueB Bdata/seq01.train.csv*
dtype0^
TextLineDataset/MatchingFilesMatchingFiles.TextLineDataset/MatchingFiles/pattern:output:0I
 TextLineDataset/compression_typeConst*
valueB B *
dtype0G
TextLineDataset/buffer_sizeConst*
valueB		 R*
dtype0	
TextLineDatasetTextLineDataset)TextLineDataset/MatchingFiles:filenames:0)TextLineDataset/compression_type:output:0$TextLineDataset/buffer_size:output:0;
SkipDataset/countConst*
value	B	 R *
dtype0	~
SkipDatasetSkipDatasetTextLineDataset:handle:0SkipDataset/count:output:0*
output_types
2*
output_shapes
: A
BatchDataset/batch_sizeConst*
value	B	 Rd*
dtype0	
BatchDatasetBatchDatasetSkipDataset:handle:0 BatchDataset/batch_size:output:0*"
output_shapes
:џџџџџџџџџ*
output_types
2O
%ParallelMapDataset/num_parallel_callsConst*
value	B :*
dtype0і
ParallelMapDatasetParallelMapDatasetBatchDataset:handle:0.ParallelMapDataset/num_parallel_calls:output:0*

Targuments
 *9
output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ* 
fR
tf_map_func_1xNaTDyIc3E*
output_types
2?
RepeatDataset/count_1Const*
value	B	 R*
dtype0	Ў
RepeatDatasetRepeatDatasetParallelMapDataset:handle:0RepeatDataset/count_1:output:0*
output_types
2*9
output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ"'
repeatdatasetRepeatDataset:handle:0

t
tf_map_func_1xNaTDyIc3E
arg0

concat
concat_125A wrapper for Defun that facilitates shape inference.A
ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0L

ExpandDims
ExpandDimsarg0ExpandDims/dim:output:0*

Tdim0*
T0L
DecodeCSV/record_defaults_0Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_1Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_2Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_3Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_4Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_5Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_6Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_7Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_8Const*
valueB*    *
dtype0L
DecodeCSV/record_defaults_9Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_10Const*
dtype0*
valueB*    M
DecodeCSV/record_defaults_11Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_12Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_13Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_14Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_15Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_16Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_17Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_18Const*
valueB*    *
dtype0M
DecodeCSV/record_defaults_19Const*
valueB*    *
dtype0
	DecodeCSV	DecodeCSVExpandDims:output:0$DecodeCSV/record_defaults_0:output:0$DecodeCSV/record_defaults_1:output:0$DecodeCSV/record_defaults_2:output:0$DecodeCSV/record_defaults_3:output:0$DecodeCSV/record_defaults_4:output:0$DecodeCSV/record_defaults_5:output:0$DecodeCSV/record_defaults_6:output:0$DecodeCSV/record_defaults_7:output:0$DecodeCSV/record_defaults_8:output:0$DecodeCSV/record_defaults_9:output:0%DecodeCSV/record_defaults_10:output:0%DecodeCSV/record_defaults_11:output:0%DecodeCSV/record_defaults_12:output:0%DecodeCSV/record_defaults_13:output:0%DecodeCSV/record_defaults_14:output:0%DecodeCSV/record_defaults_15:output:0%DecodeCSV/record_defaults_16:output:0%DecodeCSV/record_defaults_17:output:0%DecodeCSV/record_defaults_18:output:0%DecodeCSV/record_defaults_19:output:0*
na_value *
use_quote_delim(*$
OUT_TYPE
2*
field_delim,5
concat/axisConst*
value	B :*
dtype0
concat_0ConcatV2DecodeCSV:output:0DecodeCSV:output:1DecodeCSV:output:2DecodeCSV:output:3DecodeCSV:output:4DecodeCSV:output:5DecodeCSV:output:6DecodeCSV:output:7DecodeCSV:output:8DecodeCSV:output:9DecodeCSV:output:10DecodeCSV:output:11DecodeCSV:output:12DecodeCSV:output:13DecodeCSV:output:14DecodeCSV:output:15concat/axis:output:0*
T0*
N*

Tidx07
concat_1/axisConst*
value	B :*
dtype0 

concat_1_0ConcatV2DecodeCSV:output:16DecodeCSV:output:17DecodeCSV:output:18DecodeCSV:output:19concat_1/axis:output:0*

Tidx0*
T0*
N"
concatconcat_0:output:0"
concat_1concat_1_0:output:0""k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"ї
	variablesщц
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0

rnn/basic_lstm_cell/kernel:0!rnn/basic_lstm_cell/kernel/Assign!rnn/basic_lstm_cell/kernel/read:027rnn/basic_lstm_cell/kernel/Initializer/random_uniform:0

rnn/basic_lstm_cell/bias:0rnn/basic_lstm_cell/bias/Assignrnn/basic_lstm_cell/bias/read:02,rnn/basic_lstm_cell/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
Ј
!rnn/basic_lstm_cell/kernel/Adam:0&rnn/basic_lstm_cell/kernel/Adam/Assign&rnn/basic_lstm_cell/kernel/Adam/read:023rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros:0
А
#rnn/basic_lstm_cell/kernel/Adam_1:0(rnn/basic_lstm_cell/kernel/Adam_1/Assign(rnn/basic_lstm_cell/kernel/Adam_1/read:025rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros:0
 
rnn/basic_lstm_cell/bias/Adam:0$rnn/basic_lstm_cell/bias/Adam/Assign$rnn/basic_lstm_cell/bias/Adam/read:021rnn/basic_lstm_cell/bias/Adam/Initializer/zeros:0
Ј
!rnn/basic_lstm_cell/bias/Adam_1:0&rnn/basic_lstm_cell/bias/Adam_1/Assign&rnn/basic_lstm_cell/bias/Adam_1/read:023rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros:0
p
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:02%dense/kernel/Adam/Initializer/zeros:0
x
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:02'dense/kernel/Adam_1/Initializer/zeros:0
h
dense/bias/Adam:0dense/bias/Adam/Assigndense/bias/Adam/read:02#dense/bias/Adam/Initializer/zeros:0
p
dense/bias/Adam_1:0dense/bias/Adam_1/Assigndense/bias/Adam_1/read:02%dense/bias/Adam_1/Initializer/zeros:0"
trainable_variablesяь

rnn/basic_lstm_cell/kernel:0!rnn/basic_lstm_cell/kernel/Assign!rnn/basic_lstm_cell/kernel/read:027rnn/basic_lstm_cell/kernel/Initializer/random_uniform:0

rnn/basic_lstm_cell/bias:0rnn/basic_lstm_cell/bias/Assignrnn/basic_lstm_cell/bias/read:02,rnn/basic_lstm_cell/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0"(
losses

mean_squared_error/value:0"
train_op

Adam"А
metric_variables

root_mean_squared_error/total:0
root_mean_squared_error/count:0
mean_absolute_error/total:0
mean_absolute_error/count:0
mean/total:0
mean/count:0"
local_variablesэъ
 
root_mean_squared_error/total:0$root_mean_squared_error/total/Assign$root_mean_squared_error/total/read:021root_mean_squared_error/total/Initializer/zeros:0
 
root_mean_squared_error/count:0$root_mean_squared_error/count/Assign$root_mean_squared_error/count/read:021root_mean_squared_error/count/Initializer/zeros:0

mean_absolute_error/total:0 mean_absolute_error/total/Assign mean_absolute_error/total/read:02-mean_absolute_error/total/Initializer/zeros:0

mean_absolute_error/count:0 mean_absolute_error/count/Assign mean_absolute_error/count/read:02-mean_absolute_error/count/Initializer/zeros:0
T
mean/total:0mean/total/Assignmean/total/read:02mean/total/Initializer/zeros:0
T
mean/count:0mean/count/Assignmean/count/read:02mean/count/Initializer/zeros:0
P
eval_step:0eval_step/Assigneval_step/read:02eval_step/Initializer/zeros:0"
	eval_step

eval_step:0"
init_op

group_deps_1"
ready_op


concat:0"W
ready_for_local_init_op<
:
8report_uninitialized_variables_1/boolean_mask/GatherV2:0"!
local_init_op

group_deps_2"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8р8 Y5       жХ]ц	ажP%ЫжAђ*&

lossіr?


maeКD?

rmseќ7y??Ч5       жХ]ц	хДs%ЫжAђ*&

lossОв?


mae5?

rmse],Є?Lп05       жХ]ц	мВt%ЫжAђ*&

lossњиџ=


mae№z>

rmse'їД>!Ћѕ#