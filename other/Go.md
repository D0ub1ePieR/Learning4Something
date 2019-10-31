# GoLang
***
### 万物起源 hello world
```go
package main

import "fmt"

func main() {
  fmt.Println("Hello, World!")
}
```
**编译** &emsp; ``go build main.go``

**运行** &emsp; ``go run main.go`` 或在编译之后执行 ``./main``
***
### 语言结构
* 包声明  

  首行定义包名 `package main`，需要在非注释的第一行指明。每个应用程序都会有一个名为main的包。

* 引入包

  通过`import "fmt"`类似得引入所需要用的包，fmt包中包含实现了的IO函数。

* 函数

  通过`func main()`来定义函数，每一个可执行程序必须包含main函数，一般来说是第一个执行的函数*如果有init()* 则会先执行初始化。**在编写test时则不需要main，在之后提及。**

  > 注意的是 { 不能单独成行
  ```go
  func main() {
    ...   /* right */
  }

  func main()
  {
    ...   /* error */
  }
  ```

* 变量

  当标识符(包括常量、变量、类型、函数、结构字段等)以**大写字母**开头，则这个标识符的对象可以被外部包使用，*相当于public*，如果以小写字母开头，则对包外是不可见的，*相当于protected*

  变量申明,不初始化值默认为0、false、""。
  ```go
  var age int
  age = 1
  //
  var age int = 1
  //
  var age = 1
  //
  age := 1  // := 表示声明语句，只能在函数体中使用
  //
  var age int
  age := 1  // 会产生编译错误
  age, t_age := 1, 2  // 有新变量声明，不会产生编译错误
  //
  var (
    vname_1 vtype_1
    vname_2 vtype_2
    ...
  )   //可用于声明全局变量
  //
  var a = [...]int{...}   // 数组声明
  var a = [...]*int{...}  // 指针数组
  ```

  常量与特殊常量
  ```go
  const a = "abc"
  const b = len(a)    //常量表达式中，函数必须是内置函数，否则会编译不通过
  //
  const (
    i = iota  // 0
    j         // 1
    k         // 2
  )
  //
  const (
    i = iota  // 0
    j         // 1
    k = ""    // iota += 1
    l         // 3
    m = 100   // iota += 1
    n         // 5
  )
  ```

* 语句 & 表达式

  使用行分割，如果将多个语句写在同一行，需要使用 **;** 人为区分。  

  使用空表标识符 **_** 抛弃部分值
  ```go
  _, a, b = 1, 2, 3
  //  a=2, b=3
  ```

* 注释

  使用与c语言类似得注释方式。
***
### 关键词和预定义标识符

* 关键词  

|break|default|func|interface|select|
|:-:|:-:|:-:|:-:|:-:|
|**case**|**defer**|**go**|**map**|**struct**|
|**chan**|**else**|**goto**|**package**|**switch**|
|**const**|**fallthrough**|**if**|**range**|**type**|
|**continue**|**for**|**import**|**return**|**var**|

* 预定义标识符  

|append|bool|byte|cap|close|complex|complex64|complex128|uint16|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**copy**|**false**|**float32**|**float64**|**imag**|**int**|**int8**|**int16**|**uint32**|
|**int32**|**int64**|**iota**|**len**|**make**|**new**|**nil**|**panic**|**uint64**|
|**print**|**println**|**real**|**recover**|**string**|**true**|**uint**|**uint8**|**uintptr**|
***
### 值类型和引用类型

  类似c语言
  * int、float、bool、string属于值类型，变量直接指向内存中的值

  * 可以通过 **&i** 获取变量i的内存地址，即指针。空指针为nil
***
### 条件语句
> Go没有三目运算符，不支持?:形式的条件判断

```go
if expression {
  ...
} else {
  ...
}
```

* **switch语句和fallthrough**
```go
switch {
  case val1:
    ...           // 不同case之间不需要break分隔
  case val2:
    ...
  case val3:
    ...
    fallthrough   // 会强制执行下一个case，即如果进入了val3分支则val4分支也会被执行
  case val4:
    ...
  default:
    ...
}
```
* **select语句**

  [TODO]

***
### 循环语句
> 结合c与python

```go
for a := 0; a < 10; a++ {
  ...
}
//
var x = [5]int{}
for a, b := range x {
  ...         // 这里range会得到a, b为数组x中的key-value对
}
//
for true {
  ...
}
```
支持break、continue、goto语句
***
### 函数  

* 函数定义格式
```go
func function_name([parameter list]) [return_types] {
  function_body
}
```
由于存在值引用和类型引用，相关操作如交换两数就需要使用传入地址并交换值进行。

* 将函数作为参数
```go
type cb func(int) int
func main(){
  testcb(1, callback)
  testcb(2, func(x int) int{
    ...
  })
}

func testcb(x int, f cb){
  f(x)
}

func callback(x int) int {
  ...
}
```
* 闭包

* 方法  

  相当于类及其内部函数，但与C++中不同
  ```go
  type Circle struct{
    radius float64
  }
  func main(){
    var c Circle
    c.radius = 10.0
    s = c.getArea()
  }
  func (c Circle) getArea() float64 {
    return 3.14 * c.radius * c.radius
  }
  ```
***
### 切片 slice

与python数组切片类似。另外的，可以通过内置函数make初始化数组。
```go
s := make([]type, len)
```
* 声明的数组为定长数组与c语言相似，切片可以追加元素使用append、copy等操作
```go
var a = [5]int{}
b := append(a, 1)   // error, a必须为slice
//
b := append(a[:], 1)  // b=[0 0 0 0 0 1]
```
***
### Map
* 不初始化map，map会为nil，无法加入键值对

* 使用make初始化
```go
var m = map[stirng]string
m = make(map[string]string)
//
var m = map[string]string{"a":"b","c":"d"}
```
* 删除元素
```go
delete(m, "a")    //通过键的值删除元素
```
***
### 接口interface

```go
/* 定义接口 */
type interface_name interface {
   method_name1 [return_type]
   method_name2 [return_type]
   method_name3 [return_type]
   ...
   method_namen [return_type]
}

/* 定义结构体 */
type struct_name struct {
   /* variables */
}

/* 实现接口方法 */
func (struct_name_variable struct_name) method_name1() [return_type] {
   /* 方法实现 */
}
...
func (struct_name_variable struct_name) method_namen() [return_type] {
   /* 方法实现*/
}
```
***
### 并发

* goroutine  

  [TODO]
