from pygments import lex
from pygments.lexers import theorem

lean = theorem.LeanLexer()

def getTokens(code):
    return lex(code,lexer=lean)

#test
test_string = """
universes u v w

namespace algebra

open_locale tensor_product
open module

section

variables (R A : Type*) [comm_semiring R] [semiring A] [algebra R A]

/-- The multiplication in an algebra is a bilinear map.

A weaker version of this for semirings exists as `add_monoid_hom.mul`. -/
def lmul : A →ₐ[R] (End R A) :=
{ map_one' := by { ext a, exact one_mul a },
  map_mul' := by { intros a b, ext c, exact mul_assoc a b c },
  map_zero' := by { ext a, exact zero_mul a },
  commutes' := by { intro r, ext a, dsimp, rw [smul_def] },
  .. (show A →ₗ[R] A →ₗ[R] A, from linear_map.mk₂ R (*)
  (λ x y z, add_mul x y z)
  (λ c x y, by rw [smul_def, smul_def, mul_assoc _ x y])
  (λ x y z, mul_add x y z)
  (λ c x y, by rw [smul_def, smul_def, left_comm])) }
"""

if __name__ == "__main__":
    print(list(getTokens(test_string)))
    #for token in getTokens(test_string):
    #    print(token[1])

