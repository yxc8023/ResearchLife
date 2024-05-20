// 矩阵加法函数
      function matrixAdd(matrixA, matrixB) {
        if (matrixA.length !== matrixB.length || matrixA[0].length !== matrixB[0].length) {
          throw new Error('矩阵尺寸不匹配，无法进行加法运算')
        }
        const result = []
        for (let i = 0; i < matrixA.length; i++) {
          result.push([])
          for (let j = 0; j < matrixA[0].length; j++) {
            result[i].push(matrixA[i][j] + matrixB[i][j])
          }
        }
        return result
      }
      // 矩阵减法
      function matrixSubtract(matrixA, matrixB) {
        if (matrixA.length !== matrixB.length || matrixA[0].length !== matrixB[0].length) {
          throw new Error('矩阵尺寸不匹配，无法进行减法运算')
        }
        const result = []
        for (let i = 0; i < matrixA.length; i++) {
          result.push([])
          for (let j = 0; j < matrixA[0].length; j++) {
            result[i].push(matrixA[i][j] - matrixB[i][j])
          }
        }
        return result
      }
      
      // 矩阵乘法
      function matrixMultiply(matrixA, matrixB) {
        if (matrixA[0].length !== matrixB.length) {
          throw new Error('矩阵A的列数必须等于矩阵B的行数，才能进行乘法运算')
        }
        const result = []
        const rowsA = matrixA.length
        const colsB = matrixB[0].length
      
        for (let i = 0; i < rowsA; i++) {
          result.push([])
          for (let j = 0; j < colsB; j++) {
            let sum = 0
            for (let k = 0; k < matrixA[0].length; k++) {
              sum += matrixA[i][k] * matrixB[k][j]
            }
            result[i].push(sum)
          }
        }
        return result
      }
      
      //判断矩阵行列式不能为0
      function determinant(matrix) {
        // 获取矩阵的行数和列数
        const rows = matrix.length
        const cols = matrix[0].length
      
        // 验证矩阵是否为方阵
        if (rows !== cols) {
          throw new Error('输入矩阵不是方阵！')
        }
      
        // 递归计算行列式
        function calculateDeterminant(matrix) {
          // 基本情况：1x1 矩阵的行列式就是矩阵中唯一的元素
          if (matrix.length === 1) {
            return matrix[0][0]
          }
      
          let det = 0
          for (let i = 0; i < matrix.length; i++) {
            // 计算代数余子式
            const cofactor = calculateDeterminant(getCofactor(matrix, 0, i))
            // 根据行号和列号的奇偶性来确定符号
            const sign = i % 2 === 0 ? 1 : -1
            // 累加每一项的贡献
            det += sign * matrix[0][i] * cofactor
          }
          return det
        }
      
        // 获取代数余子式
        function getCofactor(matrix, row, col) {
          return matrix.filter((_, i) => i !== row).map((row) => row.filter((_, j) => j !== col))
        }
      
        // 计算矩阵的行列式
        const det = calculateDeterminant(matrix)
      
        // 判断行列式是否不等于0
        return det !== 0
      }
      //矩阵求逆
      function matrixInverse(matrix) {
        if (!determinant(matrix)) {
          throw new Error('矩阵行列式为0，不能求逆')
        }
      
        // 获取矩阵的行数和列数
        const rows = matrix.length
        const cols = matrix[0].length
      
        // 创建一个单位矩阵
        const identityMatrix = []
        for (let i = 0; i < rows; i++) {
          identityMatrix[i] = []
          for (let j = 0; j < cols; j++) {
            identityMatrix[i][j] = i === j ? 1 : 0
          }
        }
      
        // 将矩阵拼接为增广矩阵 [matrix | identityMatrix]
        const augmentedMatrix = []
        for (let i = 0; i < rows; i++) {
          augmentedMatrix[i] = matrix[i].concat(identityMatrix[i])
        }
      
        // 对增广矩阵进行行变换，将左侧矩阵转换为单位矩阵
        for (let i = 0; i < rows; i++) {
          // 如果主对角线上的元素为0，则交换行
          if (augmentedMatrix[i][i] === 0) {
            for (let j = i + 1; j < rows; j++) {
              if (augmentedMatrix[j][i] !== 0) {
                ;[augmentedMatrix[i], augmentedMatrix[j]] = [augmentedMatrix[j], augmentedMatrix[i]]
                break
              }
            }
          }
      
          // 将当前行的主对角线元素缩放为1
          const divisor = augmentedMatrix[i][i]
          for (let j = 0; j < 2 * cols; j++) {
            augmentedMatrix[i][j] /= divisor
          }
      
          // 将其他行的主对角线元素消为0
          for (let j = 0; j < rows; j++) {
            if (j !== i) {
              const factor = augmentedMatrix[j][i]
              for (let k = 0; k < 2 * cols; k++) {
                augmentedMatrix[j][k] -= factor * augmentedMatrix[i][k]
              }
            }
          }
        }
      
        // 提取增广矩阵的右侧矩阵部分，即所求的逆矩阵
        const inverseMatrix = []
        for (let i = 0; i < rows; i++) {
          inverseMatrix[i] = augmentedMatrix[i].slice(cols)
        }
      
        return inverseMatrix
      }
      
 // 最小二乘法函数
      function leastSquares(A, B) {
        var AT = numeric.transpose(A)
        var ATA = numeric.dot(AT, A)
        var ATB = numeric.dot(AT, B)
        var X = numeric.solve(ATA, ATB)
        return X
      }
	  //评价指标R2
      function calculateR2(xValues, yValues) {  
    // 确保输入数组长度相同  
			if (xValues.length !== yValues.length) {  
				throw new Error('xValues and yValues must have the same length');  
			}  
		  
			// 计算x和y的均值  
			const meanX = xValues.reduce((sum, val) => sum + val, 0) / xValues.length;  
			const meanY = yValues.reduce((sum, val) => sum + val, 0) / yValues.length;  
		  
			// 计算总平方和（TSS）  
			const totalSumOfSquares = yValues.reduce((sum, val) => sum + Math.pow(val - meanY, 2), 0);  
		  
			// 计算线性回归的参数  
			let numerator = 0;  
			let denominatorX = 0;  
			let denominatorY = 0;  
			for (let i = 0; i < xValues.length; i++) {  
				numerator += (xValues[i] - meanX) * (yValues[i] - meanY);  
				denominatorX += Math.pow(xValues[i] - meanX, 2);  
				denominatorY += Math.pow(yValues[i] - meanY, 2);  
			}  
			const slope = numerator / denominatorX;  
			const intercept = meanY - slope * meanX;  
		  
			// 计算残差平方和（RSS）  
			let residualSumOfSquares = 0;  
			for (let i = 0; i < xValues.length; i++) {  
				const predictedY = slope * xValues[i] + intercept;  
				residualSumOfSquares += Math.pow(yValues[i] - predictedY, 2);  
			}  
		  
			// 计算R²  
			const rSquared = 1 - (residualSumOfSquares / totalSumOfSquares);  
			return rSquared;  
		} 

 //保留小数位
function roundArrayToEightDecimalPlaces(coords,fraction=8) {  
    return coords.map(coord => coord.map(num => parseFloat(num.toFixed(fraction))));  
}

function dms_rad(dms) {  
    // 角度转换为弧度  
    // dms为角度  
    // 返回弧度  
    var d = Math.floor(dms);  
    var fl = (dms - d) * 100;  
    var f = Math.floor(fl);  
    var m = (fl - f) * 100;  
    f = f / 60;  
    m = m / 3600;  
    var r = (d + f + m) / 180;  
    var rad = r * Math.PI;  
    return rad;  
}  
function deg2rad(angleInDegrees) {  
    // DEG2RAD Convert angles from degrees to radians.  
    // This function converts angle units from degrees to radians for the input.  
      
    // 验证输入是否为数字  
    if (typeof angleInDegrees !== 'number' || isNaN(angleInDegrees)) {  
        throw new Error('deg2rad: Input must be a numeric value.');  
    }  
      
    // 将角度转换为弧度  
    var angleInRadians = (Math.PI / 180) * angleInDegrees;  
      
    return angleInRadians;  
}  

function get_X(B, a, b, e2) {  
    // get_X是子午线弧长计算函数  
    // 输入纬度B（弧度）  
    // 输入参数椭球长半轴a，短半轴b，第二偏心率e2  
    // 输出参数子午线弧长  
  
    // 计算偏心率e2的平方  
    var e2Squared = e2 * e2;  
  
    // 计算常数b0, b2, b4, b6, b8  
    var b0 = 1 - 3 * e2Squared / 4 + 45 * Math.pow(e2Squared, 2) / 64 - 175 * Math.pow(e2Squared, 3) / 256 + 11025 * Math.pow(e2Squared, 4) / 16384;  
    var b2 = b0 - 1;  
    var b4 = 15 * Math.pow(e2Squared, 2) / 32 - 175 * Math.pow(e2Squared, 3) / 384 + 3675 * Math.pow(e2Squared, 4) / 8192;  
    var b6 = -35 * Math.pow(e2Squared, 3) / 96 + 735 * Math.pow(e2Squared, 4) / 2048;  
    var b8 = 315 * Math.pow(e2Squared, 4) / 1024;  
  
    // 计算常数c  
    var c = Math.pow(a, 2) / b;  
  
    // 计算子午线弧长X  
var X = c * (b0 * B + (b2 * Math.cos(B) + b4 * Math.pow(Math.cos(B), 3) + b6 * Math.pow(Math.cos(B), 5) + b8 * Math.pow(Math.cos(B), 7)) * Math.sin(B));
  
    // 返回结果  
    return X;  
}
/**
 * BL2xy是将大地坐标换算成高斯坐标的函数 
 * @param {number} B - 大地坐标B
 * @param {number} L - 大地坐标L
 * @param {number} L0 - 中央子午经线L0
 * @param {number} n - 椭球编号n
 * @returns {number} - 计算得到的高斯坐标值
 */  
function BL2xy(B, L, L0, n) {  
    // BL2xy是将大地坐标换算成高斯坐标的函数  
    // 输入参量是大地坐标B, L（弧度），中央子午经线L0（弧度），椭球编号n  
    // 输出参数高斯平面坐标的自然值x，y（m）  
  
    // 椭球参数（这里假设了克拉索夫斯基椭球参数，其他椭球需要相应的参数）  
    var a = 6378137; // 长半轴  
    var b = 6356752.3142; // 短半轴  
    var e1 = Math.sqrt(a * a - b * b) / b; // 第一偏心率  
  
    // 将角度转换为弧度（如果输入已经是弧度，这一步可以省略）  
    B = deg2rad(B);  
    L = deg2rad(L);  
    L0 = deg2rad(L0);  
  
    // 调用get_X函数计算子午线弧长（需要实现这个函数）  
    var X = get_X(B, a, b, e1);  
  
    // 计算其他参数  
    var V = Math.sqrt(1 + e1 * e1 * Math.cos(B) * Math.cos(B));  
    var c = a * a / b;  
    var N = c / V;  
    var t = Math.tan(B);  
    var n = Math.sqrt(e1 * e1 * Math.cos(B) * Math.cos(B));  
    var l = L - L0;  
  
    // 计算x的近似值  
    var xp1 = X;  
    var xp2 = (N * Math.sin(B) * Math.cos(B) * l * l) / 2;  
    var xp3 = (N * Math.sin(B) * Math.pow(Math.cos(B), 3) * (5 - t * t + 9 * n * n + 4 * Math.pow(n, 4)) * Math.pow(l, 4)) / 24;  
    var xp4 = (N * Math.sin(B) * Math.pow(Math.cos(B), 5) * (61 - 58 * t * t + Math.pow(t, 4)) * Math.pow(l, 6)) / 720;  
    var x = xp1 + xp2 + xp3 + xp4;  
  
    // 计算y的近似值  
    var yp1 = N * Math.cos(B) * l;  
    var yp2 = N * Math.pow(Math.cos(B), 3) * (1 - t * t + n * n) * Math.pow(l, 3) / 6;  
    var yp3 = N * Math.pow(Math.cos(B), 5) * (5 - 18 * t * t + Math.pow(t, 4) + 14 * n * n - 58 * Math.pow(n, 2) * (t * t)) * Math.pow(l, 5) / 120;  
    var y = yp1 + yp2 + yp3 + 500000; // 这里的500000是一个常数，可能是某个特定坐标系统的偏移量  
  
    // 返回结果  
    return { x: x, y: y };  
}  


