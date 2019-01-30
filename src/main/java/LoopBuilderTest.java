import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.util.ConstantUtils;
import net.imglib2.view.Views;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.Collections;

public class LoopBuilderTest {

    public static void main(String[] args) throws Exception {
        final long[] blockSize = new long[] {100, 200, 300};
        final SparkConf conf = new SparkConf().setAppName(MethodHandles.lookup().lookupClass().getName());
        try (final JavaSparkContext sc = new JavaSparkContext(conf)) {
            final long result = sc
                    .parallelize(Collections.singletonList(1))
                    .map(one -> {
                       final RandomAccessible<UnsignedLongType>         source = ConstantUtils.constantRandomAccessible(new UnsignedLongType(1), blockSize.length);
                       final RandomAccessibleInterval<UnsignedLongType> target = ArrayImgs.unsignedLongs(blockSize);
                       Gauss3.gauss(1.0, source, target);
                       LoopBuilder.setImages(target, Views.interval(source, target)).forEachPixel(UnsignedLongType::set);
                       final UnsignedLongType sum = new UnsignedLongType(0);
                       Views.iterable(target).forEach(sum::add);
                       return sum.get();
                    })
                    .first();

            if (result != Arrays.stream(blockSize).reduce(1, (l1, l2) -> l1 * l2))
                throw new Exception("Wrong result.");

            System.out.println("Sum is " + result);
        }
    }

}
